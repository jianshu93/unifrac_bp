//! stripe_cu.rs
//! CUDA offload for Striped UniFrac (unweighted + weighted normalized only).
//!
//! New perf design (this version):
//! - GPU computes into a **device band buffer** for each assigned block-row `bi`:
//!     band is [blk x nsamp] f32 (only rows [0..bw] are meaningful for tail blocks)
//! - For a given `bi`, GPU launches kernels for all `bj >= bi` tiles writing directly into band.
//! - Only after finishing all `bj` for that `bi`, we do **one DtoH copy** of the band,
//!   then scatter to the final host distance matrix (upper + mirror).
//!
//! Multi-GPU policy:
//! - We assign work by **block-row**: worker `widx` owns all `bi` where `bi % ng == widx`.
//!   This ensures disjoint writes into host output and eliminates per-tile DtoH.
//!
//! NOTE:
//! - Requires `--features gpu` and `cudarc`.
//! - Weighted assumes normalized abundances (alpha=1).

use anyhow::{bail, Context, Result};
use bitvec::{order::Lsb0, vec::BitVec};
use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PinnedHostSlice};
use cudarc::nvrtc::compile_ptx;
use log::info;
use rayon::prelude::*;
use std::ptr::NonNull;
use std::sync::Arc;
use std::thread;
use std::time::Instant;
use cudarc::driver::PushKernelArg;

// ------------------------- raw output pointer wrapper -------------------------

#[derive(Clone, Copy)]
struct DistPtr(NonNull<f64>);

impl DistPtr {
    #[inline]
    fn as_mut_ptr(self) -> *mut f64 {
        self.0.as_ptr()
    }
}

unsafe impl Send for DistPtr {}
unsafe impl Sync for DistPtr {}

// ------------------------- options -------------------------

#[derive(Clone, Debug)]
pub struct GpuOptions {
    pub devices: Vec<usize>,
    pub block_rows: usize,
    pub block_dim_x: u32,
    pub block_dim_y: u32,

    /// Log progress every N block-rows / tiles (0 disables).
    pub progress_every: usize,
}

impl Default for GpuOptions {
    fn default() -> Self {
        Self {
            devices: Vec::new(),
            block_rows: 512,
            block_dim_x: 16,
            block_dim_y: 16,
            progress_every: 8,
        }
    }
}

// ------------------------- table modes -------------------------

pub enum InputTable<'a> {
    DenseCounts(&'a [Vec<f64>]),
    Csr {
        indptr: &'a [u32],
        indices: &'a [u32],
        data: &'a [f64],
    },
}

// ------------------------- device count -------------------------

pub fn device_count() -> Result<usize> {
    Ok(CudaContext::device_count()? as usize)
}

// ------------------------- CUDA kernels: write directly into band -------------------------

const KERNEL_SRC: &str = r#"
extern "C" __global__
void unifrac_unweighted_tile_u64_to_band(
    const unsigned long long* __restrict__ bitsA,   // [num_nodes * wordsA]
    const unsigned long long* __restrict__ bitsB,   // [num_nodes * wordsB]
    const float* __restrict__ lens,                 // [num_nodes]
    int num_nodes,
    int wordsA,
    int wordsB,
    int n,
    int i0,
    int j0,
    int bw,
    int bh,
    int only_upper,            // 1 => enforce gj > gi (diag tile)
    float* __restrict__ band   // [blk * n], row-major, ld = n
){
    int jj = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int ii = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (ii >= bw || jj >= bh) return;

    int gi = i0 + ii;
    int gj = j0 + jj;
    if (gi >= n || gj >= n) return;

    if (only_upper && gj <= gi) return;
    if (gi == gj) return;

    int wi = ii >> 6;
    int wj = jj >> 6;
    unsigned long long mi = 1ULL << (ii & 63);
    unsigned long long mj = 1ULL << (jj & 63);

    float u = 0.0f;
    float s = 0.0f;

    for (int v = 0; v < num_nodes; ++v) {
        float len = lens[v];
        if (len <= 0.0f) continue;

        unsigned long long a = bitsA[(size_t)v * (size_t)wordsA + (size_t)wi];
        unsigned long long b = bitsB[(size_t)v * (size_t)wordsB + (size_t)wj];

        int a_set = ((a & mi) != 0ULL);
        int b_set = ((b & mj) != 0ULL);

        if (a_set | b_set) {
            u += len;
            if (a_set & b_set) s += len;
        }
    }

    float d = 0.0f;
    if (u > 0.0f) d = 1.0f - (s / u);

    // band row is local ii (0..bw), col is global gj
    band[(size_t)ii * (size_t)n + (size_t)gj] = d;
}

extern "C" __global__
void unifrac_weighted_tile_f32_to_band(
    const float* __restrict__ rowsA,   // [num_nodes * bw]
    const float* __restrict__ rowsB,   // [num_nodes * bh]
    const float* __restrict__ lens,    // [num_nodes]
    int num_nodes,
    int n,
    int i0,
    int j0,
    int bw,
    int bh,
    int only_upper,
    float* __restrict__ band           // [blk * n], row-major, ld = n
){
    int jj = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int ii = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (ii >= bw || jj >= bh) return;

    int gi = i0 + ii;
    int gj = j0 + jj;
    if (gi >= n || gj >= n) return;

    if (only_upper && gj <= gi) return;
    if (gi == gj) return;

    float num = 0.0f;
    float den = 0.0f;

    for (int v = 0; v < num_nodes; ++v) {
        float len = lens[v];
        if (len <= 0.0f) continue;

        float a = rowsA[(size_t)v * (size_t)bw + (size_t)ii];
        float b = rowsB[(size_t)v * (size_t)bh + (size_t)jj];
        float s = a + b;
        if (s <= 0.0f) continue;

        float m = (a < b) ? a : b;
        den += len * s;
        num += len * (s - 2.0f * m); // len*|a-b|
    }

    float d = 0.0f;
    if (den > 0.0f) d = num / den;

    band[(size_t)ii * (size_t)n + (size_t)gj] = d;
}
"#;

// ------------------------- helpers: GPU selection -------------------------

fn auto_gpu_count(visible: usize, tiles: usize) -> usize {
    if visible == 0 {
        return 0;
    }
    if tiles <= 1 {
        return 1;
    }
    const MIN_TILES_PER_GPU: usize = 16;
    let mut want = (tiles + MIN_TILES_PER_GPU - 1) / MIN_TILES_PER_GPU;
    if want == 0 {
        want = 1;
    }
    want = want.min(tiles).min(visible);
    want.max(1)
}

fn pick_devices(opts: &GpuOptions, tiles: usize) -> Result<Vec<usize>> {
    let visible = device_count()?;
    if visible == 0 {
        return Ok(Vec::new());
    }

    if !opts.devices.is_empty() {
        for &d in &opts.devices {
            if d >= visible {
                bail!("requested CUDA device {} but only {} devices available", d, visible);
            }
        }
        return Ok(opts.devices.clone());
    }

    let use_n = auto_gpu_count(visible, tiles);
    Ok((0..use_n).collect())
}

// ------------------------- helpers: bit slicing + unions -------------------------

fn merge_union_sorted(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let mut ia = 0usize;
    let mut ib = 0usize;
    while ia < a.len() || ib < b.len() {
        match (a.get(ia), b.get(ib)) {
            (Some(&va), Some(&vb)) => {
                if va < vb {
                    out.push(va);
                    ia += 1;
                } else if vb < va {
                    out.push(vb);
                    ib += 1;
                } else {
                    out.push(va);
                    ia += 1;
                    ib += 1;
                }
            }
            (Some(&va), None) => {
                out.push(va);
                ia += 1;
            }
            (None, Some(&vb)) => {
                out.push(vb);
                ib += 1;
            }
            _ => break,
        }
    }
    out
}

/// Extract `[start..start+len)` bits from raw words into dst words (aligned).
fn extract_words_into(raw: &[u64], start_bit: usize, len_bits: usize, dst: &mut [u64]) {
    if len_bits == 0 {
        return;
    }
    let words = (len_bits + 63) >> 6;
    debug_assert_eq!(dst.len(), words);

    let bit_off = start_bit & 63;
    let w0 = start_bit >> 6;

    for w in 0..words {
        let idx = w0 + w;
        let mut v = if idx < raw.len() { raw[idx] } else { 0u64 };
        if bit_off != 0 {
            v >>= bit_off;
            if idx + 1 < raw.len() {
                v |= raw[idx + 1] << (64 - bit_off);
            }
        }
        dst[w] = v;
    }

    let tail = len_bits & 63;
    if tail != 0 {
        let mask = (1u64 << tail) - 1;
        dst[words - 1] &= mask;
    }
}

// ------------------------- scatter band to host distance matrix -------------------------

unsafe fn scatter_band_f32_to_host_f64(
    out_ptr: DistPtr,
    band: &[f32], // length = blk * nsamp (or >= bw*nsamp)
    i0: usize,
    bw: usize,
    nsamp: usize,
) {
    // Rust 2024: raw pointer ops must be inside an explicit unsafe block.
    unsafe {
        let base = out_ptr.as_mut_ptr();
        for ii in 0..bw {
            let i = i0 + ii;
            let row = &band[ii * nsamp..(ii + 1) * nsamp];
            for j in (i + 1)..nsamp {
                let d = row[j] as f64;
                *base.add(i * nsamp + j) = d;
                *base.add(j * nsamp + i) = d;
            }
        }
    }
}

// ============================================================================
//  UNWEIGHTED GPU
// ============================================================================

pub fn unifrac_striped_unweighted_gpu(
    post: &[usize],
    kids: &[Vec<usize>],
    lens: &[f32],
    leaf_ids: &[usize],
    masks: Vec<BitVec<u8, Lsb0>>,
    opts: GpuOptions,
) -> Result<Vec<f64>> {
    let t_all = Instant::now();
    let nsamp = masks.len();
    if nsamp == 0 {
        return Ok(Vec::new());
    }

    // Phase 1/2 CPU
    let (node_bits, active_per_strip, blk) =
        build_unweighted_node_bits_and_active(post, kids, lens, leaf_ids, masks)?;

    let nblk = (nsamp + blk - 1) / blk;
    let tiles_total = nblk * (nblk + 1) / 2;

    let dist = Arc::new(vec![0.0f64; nsamp * nsamp]);
    let out_ptr = DistPtr(unsafe { NonNull::new_unchecked(dist.as_ptr() as *mut f64) });

    let devices = pick_devices(&opts, tiles_total)?;
    if devices.is_empty() {
        bail!("GPU requested but no CUDA devices available");
    }
    let ng = devices.len();

    info!(
        "GPU(unweighted): nsamp={} blk={} nblk={} tiles={} visible_gpus={} using_gpus={:?}",
        nsamp,
        blk,
        nblk,
        tiles_total,
        device_count().unwrap_or(0),
        devices
    );

    // shared read-only
    let node_bits = Arc::new(node_bits);
    let active_per_strip = Arc::new(active_per_strip);
    let lens_arc = Arc::new(lens.to_vec());

    thread::scope(|scope| {
        for (widx, &dev_id) in devices.iter().enumerate() {
            let node_bits = Arc::clone(&node_bits);
            let active_per_strip = Arc::clone(&active_per_strip);
            let lens_arc = Arc::clone(&lens_arc);
            let opts = opts.clone();
            let out_ptr = out_ptr;

            scope.spawn(move || {
                let inner = || -> Result<()> {
                    let ctx = CudaContext::new(dev_id)?;
                    let stream = ctx.default_stream();

                    let ptx = compile_ptx(KERNEL_SRC)?;
                    let module = ctx.load_module(ptx)?;
                    let f_unw = module
                        .load_function("unifrac_unweighted_tile_u64_to_band")
                        .context("load kernel unifrac_unweighted_tile_u64_to_band")?;

                    // Device band buffer (blk x nsamp) and pinned host band
                    let band_elems = blk * nsamp;
                    let mut d_band: CudaSlice<f32> = stream.alloc_zeros(band_elems)?;
                    let mut h_band: PinnedHostSlice<f32> = ctx.alloc_pinned(band_elems)?;

                    let mut done_bi = 0usize;

                    // block-row ownership
                    for bi in 0..nblk {
                        if bi % ng != widx {
                            continue;
                        }

                        let i0 = bi * blk;
                        let i1 = ((bi + 1) * blk).min(nsamp);
                        let bw = i1 - i0;
                        if bw == 0 {
                            continue;
                        }

                        for bj in bi..nblk {
                            let j0 = bj * blk;
                            let j1 = ((bj + 1) * blk).min(nsamp);
                            let bh = j1 - j0;
                            if bh == 0 {
                                continue;
                            }

                            let nodes_u =
                                merge_union_sorted(&active_per_strip[bi], &active_per_strip[bj]);
                            if nodes_u.is_empty() {
                                continue;
                            }

                            let words_a = (bw + 63) >> 6;
                            let words_b = (bh + 63) >> 6;

                            let mut bits_a = vec![0u64; nodes_u.len() * words_a];
                            let mut bits_b = vec![0u64; nodes_u.len() * words_b];
                            let mut lens_v = vec![0f32; nodes_u.len()];

                            for (ni, &v) in nodes_u.iter().enumerate() {
                                lens_v[ni] = lens_arc[v];
                                let raw = node_bits[v].as_raw_slice();

                                extract_words_into(
                                    raw,
                                    i0,
                                    bw,
                                    &mut bits_a[ni * words_a..(ni + 1) * words_a],
                                );
                                extract_words_into(
                                    raw,
                                    j0,
                                    bh,
                                    &mut bits_b[ni * words_b..(ni + 1) * words_b],
                                );
                            }

                            let d_bits_a: CudaSlice<u64> = stream.clone_htod(&bits_a)?;
                            let d_bits_b: CudaSlice<u64> = stream.clone_htod(&bits_b)?;
                            let d_lens: CudaSlice<f32> = stream.clone_htod(&lens_v)?;

                            let num_nodes_i32 = nodes_u.len() as i32;
                            let words_a_i32 = words_a as i32;
                            let words_b_i32 = words_b as i32;
                            let n_i32 = nsamp as i32;

                            let i0_i32 = i0 as i32;
                            let j0_i32 = j0 as i32;
                            let bw_i32 = bw as i32;
                            let bh_i32 = bh as i32;
                            let only_upper_i32 = if bi == bj { 1i32 } else { 0i32 };

                            let cfg = LaunchConfig {
                                grid_dim: (
                                    ((bh as u32 + opts.block_dim_x - 1) / opts.block_dim_x),
                                    ((bw as u32 + opts.block_dim_y - 1) / opts.block_dim_y),
                                    1,
                                ),
                                block_dim: (opts.block_dim_x, opts.block_dim_y, 1),
                                shared_mem_bytes: 0,
                            };

                            let mut launch = stream.launch_builder(&f_unw);
                            launch.arg(&d_bits_a);
                            launch.arg(&d_bits_b);
                            launch.arg(&d_lens);
                            launch.arg(&num_nodes_i32);
                            launch.arg(&words_a_i32);
                            launch.arg(&words_b_i32);
                            launch.arg(&n_i32);
                            launch.arg(&i0_i32);
                            launch.arg(&j0_i32);
                            launch.arg(&bw_i32);
                            launch.arg(&bh_i32);
                            launch.arg(&only_upper_i32);
                            launch.arg(&mut d_band);

                            unsafe { launch.launch(cfg) }?;
                        }

                        stream.synchronize()?;
                        stream.memcpy_dtoh(&d_band, &mut h_band)?;

                        // FIX #1: as_slice() returns Result<_, DriverError> on your cudarc
                        let h_slice: &[f32] = h_band
                            .as_slice()
                            .map_err(|e| anyhow::anyhow!(e))
                            .context("PinnedHostSlice::as_slice failed (unweighted)")?;

                        unsafe { scatter_band_f32_to_host_f64(out_ptr, h_slice, i0, bw, nsamp) };

                        done_bi += 1;
                        if opts.progress_every > 0 && (done_bi % opts.progress_every == 0) {
                            info!(
                                "GPU(unweighted): dev={} worker={} finished {} block-rows",
                                dev_id, widx, done_bi
                            );
                        }
                    }

                    Ok(())
                };

                if let Err(e) = inner() {
                    panic!("GPU worker (unweighted) dev={} failed: {e:?}", dev_id);
                }
            });
        }
    });

    info!(
        "GPU(unweighted): total wall time {} ms",
        t_all.elapsed().as_millis()
    );

    Ok(Arc::try_unwrap(dist).unwrap())
}

// ============================================================================
//  WEIGHTED GPU (normalized)
// ============================================================================

pub fn unifrac_striped_weighted_gpu(
    kids: &[Vec<usize>],
    lens: &[f32],
    leaf_ids: &[usize],
    row2leaf: &[Option<usize>],
    table: InputTable<'_>,
    nsamp: usize,
    col_sums: &[f64],
    opts: GpuOptions,
) -> Result<Vec<f64>> {
    let t_all = Instant::now();
    if nsamp == 0 {
        return Ok(Vec::new());
    }
    if col_sums.len() != nsamp {
        bail!("col_sums length mismatch: got {}, expected {}", col_sums.len(), nsamp);
    }

    let total = lens.len();

    // parent[]
    let parent: Vec<usize> = {
        let mut p = vec![usize::MAX; total];
        for v in 0..total {
            for &c in &kids[v] {
                p[c] = v;
            }
        }
        p
    };

    let blk = opts
        .block_rows
        .max(1)
        .min(nsamp)
        .next_power_of_two()
        .clamp(64, 4096);
    let nblk = (nsamp + blk - 1) / blk;
    let tiles_total = nblk * (nblk + 1) / 2;

    let dist = Arc::new(vec![0.0f64; nsamp * nsamp]);
    let out_ptr = DistPtr(unsafe { NonNull::new_unchecked(dist.as_ptr() as *mut f64) });

    let devices = pick_devices(&opts, tiles_total)?;
    if devices.is_empty() {
        bail!("GPU requested but no CUDA devices available");
    }
    let ng = devices.len();

    info!(
        "GPU(weighted): nsamp={} blk={} nblk={} tiles={} visible_gpus={} using_gpus={:?}",
        nsamp,
        blk,
        nblk,
        tiles_total,
        device_count().unwrap_or(0),
        devices
    );

    // Share read-only
    let parent = Arc::new(parent);
    let lens = Arc::new(lens.to_vec());
    let leaf_ids = Arc::new(leaf_ids.to_vec());
    let row2leaf = Arc::new(row2leaf.to_vec());
    let col_sums = Arc::new(col_sums.to_vec());

    // Share table
    let dense_counts: Option<Arc<Vec<Vec<f64>>>> = match table {
        InputTable::DenseCounts(c) => Some(Arc::new(c.to_vec())),
        _ => None,
    };
    let csr_pack: Option<(Arc<Vec<u32>>, Arc<Vec<u32>>, Arc<Vec<f64>>)> = match table {
        InputTable::Csr { indptr, indices, data } => Some((
            Arc::new(indptr.to_vec()),
            Arc::new(indices.to_vec()),
            Arc::new(data.to_vec()),
        )),
        _ => None,
    };

    // Precompute stripe per block-row once
    let t_pre = Instant::now();
    let stripes: Vec<Stripe> = (0..nblk)
        .into_par_iter()
        .map(|bi| {
            let s0 = bi * blk;
            let s1 = ((bi + 1) * blk).min(nsamp);
            match (&dense_counts, &csr_pack) {
                (Some(c), _) => build_stripe_dense(
                    c,
                    &row2leaf,
                    &leaf_ids,
                    &parent,
                    &col_sums,
                    s0,
                    s1,
                    total,
                ),
                (_, Some((ip, idx, dat))) => build_stripe_csr(
                    ip,
                    idx,
                    dat,
                    &row2leaf,
                    &leaf_ids,
                    &parent,
                    &col_sums,
                    s0,
                    s1,
                    total,
                ),
                _ => panic!("invalid table mode"),
            }
        })
        .collect();
    info!(
        "GPU(weighted): precomputed {} stripes in {} ms",
        stripes.len(),
        t_pre.elapsed().as_millis()
    );
    let stripes = Arc::new(stripes);

    thread::scope(|scope| {
        for (widx, &dev_id) in devices.iter().enumerate() {
            let stripes = Arc::clone(&stripes);
            let lens = Arc::clone(&lens);
            let opts = opts.clone();
            let out_ptr = out_ptr;

            scope.spawn(move || {
                let inner = || -> Result<()> {
                    let ctx = CudaContext::new(dev_id)?;
                    let stream = ctx.default_stream();

                    let ptx = compile_ptx(KERNEL_SRC)?;
                    let module = ctx.load_module(ptx)?;
                    let f_w = module
                        .load_function("unifrac_weighted_tile_f32_to_band")
                        .context("load kernel unifrac_weighted_tile_f32_to_band")?;

                    let band_elems = blk * nsamp;
                    let mut d_band: CudaSlice<f32> = stream.alloc_zeros(band_elems)?;
                    let mut h_band: PinnedHostSlice<f32> = ctx.alloc_pinned(band_elems)?;

                    let mut done_bi = 0usize;

                    for bi in 0..nblk {
                        if bi % ng != widx {
                            continue;
                        }

                        let i0 = bi * blk;
                        let i1 = ((bi + 1) * blk).min(nsamp);
                        let bw = i1 - i0;
                        if bw == 0 {
                            continue;
                        }

                        let stripe_i = &stripes[bi];

                        for bj in bi..nblk {
                            let j0 = bj * blk;
                            let j1 = ((bj + 1) * blk).min(nsamp);
                            let bh = j1 - j0;
                            if bh == 0 {
                                continue;
                            }

                            let stripe_j = &stripes[bj];

                            let mut nodes_u = Vec::<usize>::with_capacity(
                                stripe_i.nodes.len() + stripe_j.nodes.len(),
                            );
                            nodes_u.extend_from_slice(&stripe_i.nodes);
                            nodes_u.extend_from_slice(&stripe_j.nodes);
                            nodes_u.sort_unstable();
                            nodes_u.dedup();
                            if nodes_u.is_empty() {
                                continue;
                            }

                            let mut rows_a = vec![0.0f32; nodes_u.len() * bw];
                            let mut rows_b = vec![0.0f32; nodes_u.len() * bh];
                            let mut lens_v = vec![0.0f32; nodes_u.len()];

                            for (ni, &v) in nodes_u.iter().enumerate() {
                                lens_v[ni] = lens[v];

                                if stripe_i.index[v] != u32::MAX {
                                    let ri = stripe_i.index[v] as usize;
                                    let src = &stripe_i.rows[ri];
                                    let take = bw.min(src.len());
                                    rows_a[ni * bw..ni * bw + take].copy_from_slice(&src[..take]);
                                }
                                if stripe_j.index[v] != u32::MAX {
                                    let rj = stripe_j.index[v] as usize;
                                    let src = &stripe_j.rows[rj];
                                    let take = bh.min(src.len());
                                    rows_b[ni * bh..ni * bh + take].copy_from_slice(&src[..take]);
                                }
                            }

                            let d_rows_a: CudaSlice<f32> = stream.clone_htod(&rows_a)?;
                            let d_rows_b: CudaSlice<f32> = stream.clone_htod(&rows_b)?;
                            let d_lens: CudaSlice<f32> = stream.clone_htod(&lens_v)?;

                            let num_nodes_i32 = nodes_u.len() as i32;
                            let n_i32 = nsamp as i32;
                            let i0_i32 = i0 as i32;
                            let j0_i32 = j0 as i32;
                            let bw_i32 = bw as i32;
                            let bh_i32 = bh as i32;
                            let only_upper_i32 = if bi == bj { 1i32 } else { 0i32 };

                            let cfg = LaunchConfig {
                                grid_dim: (
                                    ((bh as u32 + opts.block_dim_x - 1) / opts.block_dim_x),
                                    ((bw as u32 + opts.block_dim_y - 1) / opts.block_dim_y),
                                    1,
                                ),
                                block_dim: (opts.block_dim_x, opts.block_dim_y, 1),
                                shared_mem_bytes: 0,
                            };

                            let mut launch = stream.launch_builder(&f_w);
                            launch.arg(&d_rows_a);
                            launch.arg(&d_rows_b);
                            launch.arg(&d_lens);
                            launch.arg(&num_nodes_i32);
                            launch.arg(&n_i32);
                            launch.arg(&i0_i32);
                            launch.arg(&j0_i32);
                            launch.arg(&bw_i32);
                            launch.arg(&bh_i32);
                            launch.arg(&only_upper_i32);
                            launch.arg(&mut d_band);
                            unsafe { launch.launch(cfg) }?;
                        }

                        stream.synchronize()?;
                        stream.memcpy_dtoh(&d_band, &mut h_band)?;

                        // FIX #1 again
                        let h_slice: &[f32] = h_band
                            .as_slice()
                            .map_err(|e| anyhow::anyhow!(e))
                            .context("PinnedHostSlice::as_slice failed (weighted)")?;

                        unsafe { scatter_band_f32_to_host_f64(out_ptr, h_slice, i0, bw, nsamp) };

                        done_bi += 1;
                        if opts.progress_every > 0 && (done_bi % opts.progress_every == 0) {
                            info!(
                                "GPU(weighted): dev={} worker={} finished {} block-rows",
                                dev_id, widx, done_bi
                            );
                        }
                    }

                    Ok(())
                };

                if let Err(e) = inner() {
                    panic!("GPU worker (weighted) dev={} failed: {e:?}", dev_id);
                }
            });
        }
    });

    info!(
        "GPU(weighted): total wall time {} ms",
        t_all.elapsed().as_millis()
    );

    Ok(Arc::try_unwrap(dist).unwrap())
}

// ============================================================================
//  CPU PHASES + STRIPE BUILDING (unchanged logic)
// ============================================================================

fn build_unweighted_node_bits_and_active(
    post: &[usize],
    kids: &[Vec<usize>],
    lens: &[f32],
    leaf_ids: &[usize],
    mut masks: Vec<BitVec<u8, Lsb0>>,
) -> Result<(Vec<BitVec<u64, Lsb0>>, Vec<Vec<usize>>, usize)> {
    let nsamp = masks.len();
    let total = lens.len();
    let n_threads = rayon::current_num_threads().max(1);

    let stripe = (nsamp + n_threads - 1) / n_threads;
    let words_str = (stripe + 63) >> 6;

    let mut node_masks: Vec<Vec<Vec<u64>>> = (0..n_threads)
        .map(|_| vec![vec![0u64; words_str]; total])
        .collect();

    let t0 = Instant::now();
    rayon::scope(|scope| {
        for (tid, node_masks_t) in node_masks.iter_mut().enumerate() {
            let stripe_start = tid * stripe;
            if stripe_start >= nsamp {
                break;
            }
            let stripe_end = (stripe_start + stripe).min(nsamp);

            let masks_slice = &masks[stripe_start..stripe_end];
            let leaf = leaf_ids;
            let kids = kids;
            let post = post;

            scope.spawn(move |_| {
                for (local_s, sm) in masks_slice.iter().enumerate() {
                    for pos in sm.iter_ones() {
                        let v = leaf[pos];
                        let w = local_s >> 6;
                        let b = local_s & 63;
                        node_masks_t[v][w] |= 1u64 << b;
                    }
                }
                for &v in post {
                    for &c in &kids[v] {
                        for w in 0..words_str {
                            node_masks_t[v][w] |= node_masks_t[c][w];
                        }
                    }
                }
            });
        }
    });
    info!(
        "GPU(unweighted): phase-1 masks built {} ms",
        t0.elapsed().as_millis()
    );

    masks.clear();
    masks.shrink_to_fit();

    let mut node_bits: Vec<BitVec<u64, Lsb0>> =
        (0..total).map(|_| BitVec::repeat(false, nsamp)).collect();

    node_bits
        .par_iter_mut()
        .enumerate()
        .for_each(|(v, bv)| {
            let dst_words = bv.as_raw_mut_slice();
            for tid in 0..n_threads {
                let stripe_start = tid * stripe;
                let stripe_end = (stripe_start + stripe).min(nsamp);
                if stripe_start >= stripe_end {
                    break;
                }

                let src_words = &node_masks[tid][v];
                let word_off = stripe_start >> 6;
                let bit_off = (stripe_start & 63) as u32;

                for w in 0..src_words.len() {
                    let mut val = src_words[w];
                    if w == src_words.len() - 1 {
                        let tail_bits = (stripe_end - stripe_start) & 63;
                        if tail_bits != 0 {
                            val &= (1u64 << tail_bits) - 1;
                        }
                    }
                    if val == 0 {
                        continue;
                    }
                    dst_words[word_off + w] |= val << bit_off;
                    if bit_off != 0 && word_off + w + 1 < dst_words.len() {
                        dst_words[word_off + w + 1] |= val >> (64 - bit_off);
                    }
                }
            }
        });

    drop(node_masks);

    let n_threads2 = rayon::current_num_threads().max(1);
    let est_blk = ((nsamp as f64 / (2.0 * n_threads2 as f64)).sqrt()) as usize;
    let blk = est_blk.clamp(64, 512).next_power_of_two();
    let nblk = (nsamp + blk - 1) / blk;

    let mut active_per_strip: Vec<Vec<usize>> = vec![Vec::new(); nblk];
    for v in 0..total {
        if lens[v] <= 0.0 {
            continue;
        }
        let raw = node_bits[v].as_raw_slice();
        for bi in 0..nblk {
            let i0 = bi * blk;
            let i1 = ((bi + 1) * blk).min(nsamp);
            let w0 = i0 >> 6;
            let w1 = (i1 + 63) >> 6;
            if raw[w0..w1].iter().any(|&w| w != 0) {
                active_per_strip[bi].push(v);
            }
        }
    }

    for lst in &mut active_per_strip {
        lst.sort_unstable();
        lst.dedup();
    }

    info!(
        "GPU(unweighted): phase-2 active lists built (blk={}, nblk={})",
        blk, nblk
    );

    Ok((node_bits, active_per_strip, blk))
}

#[derive(Clone)]
struct Stripe {
    nodes: Vec<usize>,
    rows: Vec<Vec<f32>>,
    index: Vec<u32>,
    width: usize,
}

fn ensure_row_slot<'a>(
    v: usize,
    idx_of: &mut [u32],
    nodes: &mut Vec<usize>,
    rows: &'a mut Vec<Vec<f32>>,
    width: usize,
) -> (usize, &'a mut [f32]) {
    let idx = idx_of[v];
    if idx != u32::MAX {
        let i = idx as usize;
        return (i, rows[i].as_mut_slice());
    }
    let new_idx = rows.len() as u32;
    idx_of[v] = new_idx;
    nodes.push(v);
    rows.push(vec![0f32; width]);
    let i = new_idx as usize;
    (i, rows[i].as_mut_slice())
}

fn build_stripe_dense(
    counts: &[Vec<f64>],
    row2leaf: &[Option<usize>],
    leaf_ids: &[usize],
    parent: &[usize],
    col_sums: &[f64],
    s0: usize,
    s1: usize,
    total: usize,
) -> Stripe {
    let width = s1 - s0;
    let mut idx_of = vec![u32::MAX; total];
    let mut nodes: Vec<usize> = Vec::new();
    let mut rows: Vec<Vec<f32>> = Vec::new();

    for (r, lopt) in row2leaf.iter().enumerate() {
        let Some(lp) = *lopt else { continue };
        let v_leaf = leaf_ids[lp];

        for s in s0..s1 {
            let denom = col_sums[s];
            if denom <= 0.0 {
                continue;
            }
            let val = counts[r][s];
            if val <= 0.0 {
                continue;
            }
            let inc = (val / denom) as f32;

            let mut v = v_leaf;
            loop {
                let (_row_idx, row) = ensure_row_slot(v, &mut idx_of, &mut nodes, &mut rows, width);
                row[s - s0] += inc;
                let p = parent[v];
                if p == usize::MAX {
                    break;
                }
                v = p;
            }
        }
    }

    Stripe { nodes, rows, index: idx_of, width }
}

fn build_stripe_csr(
    indptr: &[u32],
    indices: &[u32],
    data: &[f64],
    row2leaf: &[Option<usize>],
    leaf_ids: &[usize],
    parent: &[usize],
    col_sums: &[f64],
    s0: usize,
    s1: usize,
    total: usize,
) -> Stripe {
    let width = s1 - s0;
    let mut idx_of = vec![u32::MAX; total];
    let mut nodes: Vec<usize> = Vec::new();
    let mut rows: Vec<Vec<f32>> = Vec::new();

    for r in 0..row2leaf.len() {
        let Some(lp) = row2leaf[r] else { continue };
        let v_leaf = leaf_ids[lp];

        let a = indptr[r] as usize;
        let b = indptr[r + 1] as usize;
        for k in a..b {
            let s = indices[k] as usize;
            if s < s0 || s >= s1 {
                continue;
            }
            let denom = col_sums[s];
            if denom <= 0.0 {
                continue;
            }
            let val = data[k];
            if val <= 0.0 {
                continue;
            }
            let inc = (val / denom) as f32;

            let mut v = v_leaf;
            loop {
                let (_row_idx, row) = ensure_row_slot(v, &mut idx_of, &mut nodes, &mut rows, width);
                row[s - s0] += inc;
                let p = parent[v];
                if p == usize::MAX {
                    break;
                }
                v = p;
            }
        }
    }

    Stripe { nodes, rows, index: idx_of, width }
}