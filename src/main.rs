//! succinct-BP UniFrac  (unweighted)
//!  * default  – pairwise, parallel rows
//!  * --striped – single post-order pass, **parallel blocks**
//!      (works for tens-of-thousands samples)
use std::time::Instant;
use log::info;
use anyhow::{Context, Result};
use bitvec::{order::Lsb0, vec::BitVec};
use clap::{Arg, ArgAction, Command};
use env_logger;
use rayon::prelude::*;
use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, Write},
    sync::Arc,
};

use newick::{one_from_filename, Newick, NodeID};
use succparen::{
    bitwise::{ops::NndOne, SparseOneNnd},
    tree::{
        balanced_parens::{BalancedParensTree, Node as BpNode},
        traversal::{DepthFirstTraverse, VisitNode},
        LabelVec,
    },
};
use succparen::tree::Node;

use std::ptr::NonNull;

// Plain new-type – automatically `Copy`.
#[derive(Clone, Copy)]
struct DistPtr(NonNull<f64>);

//  SAFETY: we guarantee in our algorithm that every thread writes a
//  *disjoint* rectangle of the matrix.  No two threads touch the same
//  element → pointer can be shared.
unsafe impl Send for DistPtr {}
unsafe impl Sync for DistPtr {}

type NwkTree = newick::NewickTree;

struct SuccTrav<'a> {
    t: &'a NwkTree,
    stack: Vec<(NodeID, usize, usize)>,
    lens: &'a mut Vec<f32>,
}
impl<'a> SuccTrav<'a> {
    fn new(t: &'a NwkTree, lens: &'a mut Vec<f32>) -> Self {
        Self { t, stack: vec![(t.root(), 0, 0)], lens }
    }
}
impl<'a> DepthFirstTraverse for SuccTrav<'a> {
    type Label = ();

    fn next(&mut self) -> Option<VisitNode<Self::Label>> {
        let (id, lvl, nth) = self.stack.pop()?;
        let n_children = self.t[id].children().len();
        for (k, &c) in self.t[id].children().iter().enumerate().rev() {
            let nth = n_children - 1 - k;
            self.stack.push((c, lvl + 1, nth));
        }
        if self.lens.len() <= id {
            self.lens.resize(id + 1, 0.0);
        }
        self.lens[id] = self.t[id].branch().copied().unwrap_or(0.0);
        Some(VisitNode::new((), lvl, nth))
    }
}

fn collect_children<N: NndOne>(
    node: &BpNode<LabelVec<()>, N, &BalancedParensTree<LabelVec<()>, N>>,
    kids: &mut [Vec<usize>],
    post: &mut Vec<usize>,
) {
    let pid = node.id() as usize;
    for edge in node.children() {
        let cid = edge.node.id() as usize;
        kids[pid].push(cid);
        collect_children(&edge.node, kids, post);
    }
    post.push(pid);
}

fn read_table(p: &str) -> Result<(Vec<String>, Vec<String>, Vec<Vec<f64>>)> {
    let f = File::open(p)?;
    let mut lines = BufReader::new(f).lines();
    let hdr = lines.next().context("empty table")??;
    let mut it = hdr.split('\t');
    it.next();
    let samples = it.map(|s| s.to_owned()).collect();

    let mut taxa = Vec::new();
    let mut mat  = Vec::new();
    for l in lines {
        let row = l?;
        let mut p = row.split('\t');
        let tax = p.next().unwrap().to_owned();
        let vals = p
            .map(|v| if v.parse::<f64>().unwrap_or(0.0) > 0.0 { 1.0 } else { 0.0 })
            .collect();
        taxa.push(tax);
        mat.push(vals);
    }
    Ok((taxa, samples, mat))
}

fn write_matrix(names: &[String], d: &[f64], n: usize, p: &str) -> Result<()> {
    let mut f = File::create(p)?;
    write!(f, "Sample")?;
    for s in names { write!(f, "\t{s}")?; }
    writeln!(f)?;
    for i in 0..n {
        write!(f, "{}", names[i])?;
        for j in 0..n {
            write!(f, "\t{:.7}", d[i * n + j])?;
        }
        writeln!(f)?;
    }
    Ok(())
}

//  Pair-wise algorithm
fn unifrac_pair(
    post: &[usize],
    kids: &[Vec<usize>],
    lens: &[f32],
    leaf_ids: &[usize],
    a: &BitVec<u8, Lsb0>,
    b: &BitVec<u8, Lsb0>,
) -> f64
{
    const A_BIT: u8 = 0b01;
    const B_BIT: u8 = 0b10;
    //  0 .. total-1
    let mut mask = vec![0u8; lens.len()];            
    // leaves
    for (leaf_pos, &nid) in leaf_ids.iter().enumerate() {
        if a[leaf_pos] { mask[nid] |= A_BIT; }
        if b[leaf_pos] { mask[nid] |= B_BIT; }
    }
    // internal nodes – post-order ⇒ children processed already
    for &v in post {
        for &c in &kids[v] {
            mask[v] |= mask[c];
        }
    }

    // Step 2: accumulate only where at least one present
    let (mut shared, mut union) = (0.0, 0.0);

    for &v in post {
        let m = mask[v];
        // Guard, only taxa showed up in at least one sample wil be used
        if m == 0 { continue }                       

        let len = lens[v] as f64;
        // exactly one present?
        if m == A_BIT || m == B_BIT { union  += len; }
        // m == 0b11 
        else { shared += len; union += len; }
    }

    if union == 0.0 { 0.0 } else { 1.0 - shared / union }
}

/// Striped UniFrac (unweighted) with *sparse-node* optimisation.
/// ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
fn unifrac_striped_par(
    post:     &[usize],              // post-order list of *all* internal nodes
    kids:     &[Vec<usize>],         // children per node  (for completeness – not used here)
    lens:     &[f32],                // branch length  **f32 → already compact**
    leaf_ids: &[usize],              // mapping leaf-index → node-id
    masks:    &[BitVec<u8, Lsb0>],   // presence/absence per sample
) -> Vec<f64>
{
    /* ─────────────────────  constants & helpers  ───────────────────── */
    let nsamp   = masks.len();
    let total   = lens.len();
    let threads = rayon::current_num_threads().max(1);
    let mut node_masks: Vec<BitVec<u8, Lsb0>> =
        (0..total).map(|_| BitVec::repeat(false, nsamp)).collect();
    /* ─────────── Phase-1: propagate leaf presence up the tree ───────── */
    let t0 = Instant::now();

    /* leaves – unchanged */
    for (leaf_pos, &nid) in leaf_ids.iter().enumerate() {
        for (s, sm) in masks.iter().enumerate() {
            if sm[leaf_pos] { node_masks[nid].set(s, true); }
        }
    }

    /* internal nodes */
    for &v in post {
        let mut acc = node_masks[v].clone();      // start with own mask
        for &c in &kids[v] {
            acc |= &node_masks[c];                // immutable borrow finished here
        }
        node_masks[v] = acc;                      // single mutable write-back
    }
    info!("phase-1 masks built {:>6} ms", t0.elapsed().as_millis());

    /* ─────────── Phase-2: matrix tiling & sparse node lists ─────────── */
    let est_blk = ((nsamp as f64 / (2.0 * threads as f64)).sqrt()) as usize;
    let blk     = est_blk.clamp(64, 512).next_power_of_two();     //   64…512  (power-of-2)
    let nblk    = (nsamp + blk - 1) / blk;                        //   #tiles  per axis

    // build *once* – for every strip (tile-row/col) store nodes that have
    // *any* sample present in that strip.  Cost:  (#nodes × nblk) bit tests.
    let mut active_per_strip: Vec<Vec<usize>> = vec![Vec::new(); nblk];

    for v in 0..total {
        let bits = &node_masks[v];
        let raw  = bits.as_raw_slice();            // &[u64], little-endian words
        // iterate strips
        for bi in 0..nblk {
            let i0 = bi * blk;
            let i1 = ((bi + 1) * blk).min(nsamp);
            // word range covering that strip
            if bits[i0..i1].any() {          // scans only O(strip-size) bits,
                active_per_strip[bi].push(v) // but strip ≤ 512 ⇒ negligible cost
            }
        }
    }
    // post-order guarantees ascending node id – active lists are already sorted
    info!("phase-2 sparse lists built ({} strips)", nblk);

    /* ─────────── Phase-3: parallel block sweep (sparse) ─────────────── */
    let dist   = Arc::new(vec![0.0f64; nsamp * nsamp]);
    let d_ptr  = DistPtr(unsafe { NonNull::new_unchecked(dist.as_ptr() as *mut f64) });

    let pairs: Vec<(usize, usize)> =
        (0..nblk).flat_map(|bi| (bi..nblk).map(move |bj| (bi, bj))).collect();

    let t3 = Instant::now();
    pairs.into_par_iter().for_each(move |(bi, bj)| {
        /* rectangle coordinates */
        let d_ptr = d_ptr;  
        let (i0, i1) = (bi * blk, ((bi + 1) * blk).min(nsamp));
        let (j0, j1) = (bj * blk, ((bj + 1) * blk).min(nsamp));
        let bw = i1 - i0;
        let bh = j1 - j0;

        /* block-local accumulators */
        let mut union   = vec![0.0f64; bw * bh];
        let mut shared  = vec![0.0f64; bw * bh];

        /* merge the two *sorted* active-node lists on the fly */
        let list_a = &active_per_strip[bi];
        let list_b = &active_per_strip[bj];
        let mut ia = 0;
        let mut ib = 0;

        while ia < list_a.len() || ib < list_b.len() {
            let v = match (list_a.get(ia), list_b.get(ib)) {
                (Some(&va), Some(&vb)) => { if va < vb { ia += 1; va }
                                            else if vb < va { ib += 1; vb }
                                            else { ia += 1; ib += 1; va } }
                (Some(&va), None)   => { ia += 1; va },
                (None,   Some(&vb)) => { ib += 1; vb },
                _ => unreachable!(),
            };

            let len  = lens[v] as f64;
            let bits = &node_masks[v];

            for (ii, i) in (i0..i1).enumerate() {
                let a = bits[i];                         // sample-i present?
                for (jj, j) in (j0..j1).enumerate() {
                    if j <= i { continue; }
                    let b = bits[j];                     // sample-j present?
                    if !(a || b) { continue; }           // nothing to add
                    let idx = ii * bh + jj;
                    union [idx]  += len;                 // a ∨ b
                    if a && b { shared[idx] += len; }    // a ∧ b
                }
            }
        }

        /* write-back (rectangle is disjoint → race-free) */
        unsafe {
            let base = d_ptr.0.as_ptr();
            for (ii, i) in (i0..i1).enumerate() {
                for (jj, j) in (j0..j1).enumerate() {
                    if j <= i { continue; }
                    let idx = ii * bh + jj;
                    let u   = union [idx];
                    let s   = shared[idx];
                    let d   = if u == 0.0 { 0.0 } else { 1.0 - s / u };
                    *base.add(i * nsamp + j) = d;
                    *base.add(j * nsamp + i) = d;
                }
            }
        }
    });
    info!("phase-3 block pass {:>6} ms", t3.elapsed().as_millis());

    Arc::try_unwrap(dist).unwrap()
}

fn main() -> Result<()> {
    env_logger::Builder::from_default_env().init();

    let m = Command::new("unifrac-rs")
        .arg(
            Arg::new("tree")
            .short('t')
            .long("tree")
            .help("Input tree in Newick format")
            .required(true))
        .arg(
            Arg::new("input")
            .short('i')
            .long("input")
            .help("Input OTU table in TSV format")
            .required(true))
        .arg(
            Arg::new("output")
            .short('o')
            .long("output")
            .help("Output distance matrix in TSV format")
            .default_value("unifrac.tsv"))
        .arg(
            Arg::new("striped")
            .long("striped")
            .help("Use striped UniFrac algorithm")
            .action(ArgAction::SetTrue),
        )
        .get_matches();

    let striped = m.get_flag("striped");
    let tree_file = m.get_one::<String>("tree").unwrap();
    let tbl_file = m.get_one::<String>("input").unwrap();
    let out_file = m.get_one::<String>("output").unwrap();
    

    rayon::ThreadPoolBuilder::new()
    .num_threads(num_cpus::get())   // use every logical core, all threads
    .build_global()
    .unwrap();

    // build tree in balanced parentheses format
    let t: NwkTree = one_from_filename(tree_file).context("parse newick")?;
    let mut lens = Vec::<f32>::new();
    let trav = SuccTrav::new(&t, &mut lens);
    let bp: BalancedParensTree<LabelVec<()>, SparseOneNnd> =
        BalancedParensTree::new_builder(trav, LabelVec::<()>::new()).build_all();

    /* leaves */
    let mut leaf_ids = Vec::<usize>::new();
    let mut leaf_nm  = Vec::<String>::new();
    for n in t.nodes() {
        if t[n].is_leaf() {
            leaf_ids.push(n);
            leaf_nm.push(
                t.name(n).map(ToOwned::to_owned).unwrap_or_else(|| format!("L{n}")),
            );
        }
    }

    // children and post-order tranversal
    let total = bp.len() + 1;
    lens.resize(total, 0.0);
    let mut kids = vec![Vec::<usize>::new(); total];
    let mut post = Vec::<usize>::with_capacity(total);
    collect_children::<SparseOneNnd>(&bp.root(), &mut kids, &mut post);

    // parsing OTU table
    // taxa, samples, presence/absence matrix
    // taxa[i][j] is presence of taxon i in sample j
    // taxa[i] is the taxon name
    // samples[j] is the sample name
    // pres[i][j] is 1 if taxon i is present in sample j
    // (i.e., has at least one leaf in the subtree of taxon i)
    // pres[i][j] is 0 if taxon i is absent in sample j
    // (i.e., has no leaves in the subtree of taxon i)
    let (taxa, samples, pres) = read_table(tbl_file)?;
    let nsamp = samples.len();

    let t2leaf: HashMap<&str, usize> = leaf_nm
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();

    let mut masks: Vec<BitVec<u8, Lsb0>> =
        (0..nsamp).map(|_| BitVec::repeat(false, leaf_ids.len())).collect();

    for (ti, tax) in taxa.iter().enumerate() {
        if let Some(&leaf) = t2leaf.get(tax.as_str()) {
            for (s, bits) in masks.iter_mut().enumerate() {
                if pres[ti][s] > 0.0 {
                    bits.set(leaf, true);
                }
            }
        }
    }
    // pairwise or striped UniFrac interface
    let dist = if striped {
        unifrac_striped_par(&post, &kids, &lens, &leaf_ids, &masks)
    } else {
        (0..nsamp)
            .into_par_iter()
            .map(|i| {
                let mut row = vec![0.0f64; nsamp];
                for j in i + 1..nsamp {
                    row[j] = unifrac_pair(&post, &kids, &lens, &leaf_ids, &masks[i], &masks[j]);
                }
                (i, row)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .fold(vec![0.0f64; nsamp * nsamp], |mut acc, (i, row)| {
                for (j, &v) in row.iter().enumerate() {
                    if v != 0.0 {
                        acc[i * nsamp + j] = v;
                        acc[j * nsamp + i] = v;
                    }
                }
                acc
            })
    };
    write_matrix(&samples, &dist, nsamp, out_file)
}
