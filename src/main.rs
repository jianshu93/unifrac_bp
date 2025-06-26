//! succinct-BP UniFrac  (unweighted)
//!  * default  – pair-at-a-time, parallel rows
//!  * --striped – single post-order pass, **parallel blocks**
//!      (works for tens-of-thousands samples)

use anyhow::{Context, Result};
use bitvec::{order::Lsb0, vec::BitVec};
use clap::{Arg, ArgAction, Command};
use env_logger;
use rayon::prelude::*;
use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, Write},
    sync::{Arc, Mutex},
};

use newick::{one_from_filename, Newick, NodeID};
use succ::{
    bitwise::{ops::NndOne, SparseOneNnd},
    tree::{
        balanced_parens::{BalancedParensTree, Node as BpNode},
        traversal::{DepthFirstTraverse, VisitNode},
        LabelVec,
    },
};
use succ::tree::Node;
/* -------------------------------------------------------------------------- */
/*  Balanced-parentheses construction                                         */
/* -------------------------------------------------------------------------- */

type NwkTree = newick::NewickTree;

struct SuccTrav<'a> {
    t:     &'a NwkTree,
    stack: Vec<(NodeID, usize, usize)>,
    lens:  &'a mut Vec<f32>,
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

/* -------------------------------------------------------------------------- */
/*  OTU table                                                                 */
/* -------------------------------------------------------------------------- */

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

/* -------------------------------------------------------------------------- */
/*  Pair-wise algorithm (row-parallel)                                        */
/* -------------------------------------------------------------------------- */

fn unifrac_pair(
    post: &[usize],
    kids: &[Vec<usize>],
    lens: &[f32],
    leaf_ids: &[usize],
    a: &BitVec<u8, Lsb0>,
    b: &BitVec<u8, Lsb0>,
) -> f64 {
    let mut in_a = vec![0u8; lens.len()];
    let mut in_b = vec![0u8; lens.len()];
    for (k, &nid) in leaf_ids.iter().enumerate() {
        if a[k] { in_a[nid] = 1; }
        if b[k] { in_b[nid] = 1; }
    }
    let (mut shared, mut union) = (0.0, 0.0);
    for &v in post {
        for &c in &kids[v] {
            in_a[v] |= in_a[c];
            in_b[v] |= in_b[c];
        }
        let len = lens[v] as f64;
        let a1  = in_a[v] == 1;
        let b1  = in_b[v] == 1;
        if a1 || b1 { union  += len; }
        if a1 && b1 { shared += len; }
    }
    if union == 0.0 { 0.0 } else { 1.0 - shared / union }
}

/* -------------------------------------------------------------------------- */
/*  Striped algorithm (parallel blocks)                                       */
/* -------------------------------------------------------------------------- */

const BLK: usize = 128;               // block size (power-of-2 helps cache)

fn unifrac_striped_par(
    post:     &[usize],
    kids:     &[Vec<usize>],
    lens:     &[f32],
    leaf_ids: &[usize],
    masks:    &[BitVec<u8, Lsb0>],
) -> Vec<f64> {
    let nsamp = masks.len();
    let total = lens.len();

    /* 1) node → samples presence (transposed) -------------------------- */
    let mut node_masks: Vec<BitVec<u8, Lsb0>> =
        (0..total).map(|_| BitVec::repeat(false, nsamp)).collect();

    for (leaf_pos, &nid) in leaf_ids.iter().enumerate() {
        for (s, sm) in masks.iter().enumerate() {
            if sm[leaf_pos] {
                node_masks[nid].set(s, true);
            }
        }
    }
    for &v in post {
        for &c in &kids[v] {
            let child_mask  = node_masks[c].clone();
            node_masks[v]  |= &child_mask;
        }
    }

    /* 2) distance matrix skeleton ------------------------------------- */
    let dist   = Arc::new(Mutex::new(vec![0.0f64; nsamp * nsamp]));
    let nblk   = (nsamp + BLK - 1) / BLK;

    /* 3) parallel over block pairs ------------------------------------ */
    (0..nblk).into_par_iter().for_each(|bi| {
        let i0 = bi * BLK;
        let i1 = (i0 + BLK).min(nsamp);

        for bj in bi..nblk {
            let j0 = bj * BLK;
            let j1 = (j0 + BLK).min(nsamp);

            let b_w = i1 - i0;
            let b_h = j1 - j0;
            let mut union  = vec![0f64; b_w * b_h];
            let mut shared = vec![0f64; b_w * b_h];

            /* accumulate union/shared for this block pair */
            for &v in post {
                let len = lens[v] as f64;
                let bv  = &node_masks[v];

                for (ii, i) in (i0..i1).enumerate() {
                    let a = bv[i];
                    for (jj, j) in (j0..j1).enumerate() {
                        if j <= i { continue }           // upper tri only
                        let b = bv[j];
                        let idx = ii * b_h + jj;
                        if a || b { union [idx] += len; }
                        if a && b { shared[idx] += len; }
                    }
                }
            }

            /* convert to distance & write back */
            let mut dlock = dist.lock().unwrap();
            for (ii, i) in (i0..i1).enumerate() {
                for (jj, j) in (j0..j1).enumerate() {
                    if j <= i { continue }
                    let idx_blk = ii * b_h + jj;
                    let u = union [idx_blk];
                    let s = shared[idx_blk];
                    let v = if u == 0.0 { 0.0 } else { 1.0 - s / u };
                    dlock[i * nsamp + j] = v;
                    dlock[j * nsamp + i] = v;
                }
            }
        }
    });

    Arc::into_inner(dist).unwrap().into_inner().unwrap()
}

/* -------------------------------------------------------------------------- */
/*  Main                                                                      */
/* -------------------------------------------------------------------------- */

fn main() -> Result<()> {
    env_logger::Builder::from_default_env().init();

    let m = Command::new("unifrac_succ")
        .arg(Arg::new("tree").short('t').required(true))
        .arg(Arg::new("input").short('i').required(true))
        .arg(Arg::new("output").short('o').required(true))
        .arg(
            Arg::new("striped")
                .long("striped")
                .help("use striped all-pairs algorithm (parallel blocks)")
                .action(ArgAction::SetTrue),
        )
        .get_matches();

    let striped   = m.get_flag("striped");
    let tree_file = m.get_one::<String>("tree").unwrap();
    let tbl_file  = m.get_one::<String>("input").unwrap();
    let out_file  = m.get_one::<String>("output").unwrap();

    /* --- tree → balanced parentheses ---------------------------------- */
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

    /* children + post-order */
    let total = bp.len() + 1;
    lens.resize(total, 0.0);
    let mut kids = vec![Vec::<usize>::new(); total];
    let mut post = Vec::<usize>::with_capacity(total);
    collect_children::<SparseOneNnd>(&bp.root(), &mut kids, &mut post);

    /* --- OTU table ------------------------------------------------------ */
    let (taxa, samples, pres) = read_table(tbl_file)?;
    let nsamp = samples.len();

    let t2leaf: HashMap<&str, usize> =
        leaf_nm.iter().map(|n| (n.as_str(), *n2index(n, &leaf_nm))).collect();

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

    /* --- UniFrac -------------------------------------------------------- */
    let dist = if striped {
        unifrac_striped_par(&post, &kids, &lens, &leaf_ids, &masks)
    } else {
        (0..nsamp)
            .into_par_iter()
            .map(|i| {
                let mut row = vec![0.0f64; nsamp];
                for j in i + 1..nsamp {
                    row[j] = unifrac_pair(
                        &post, &kids, &lens, &leaf_ids, &masks[i], &masks[j]);
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

    /* --- output --------------------------------------------------------- */
    write_matrix(&samples, &dist, nsamp, out_file)
}

/* helper for leaf-name → index when building the hash-map */
fn n2index<'a>(name: &'a str, all: &'a [String]) -> &'a usize {
    all.iter()
        .position(|s| s == name)
        .as_ref()
        .map(|i| unsafe { &*(i as *const usize) })
        .unwrap()
}