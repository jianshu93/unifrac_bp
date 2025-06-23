//!  succinct-BP unweighted UniFrac  (succ + newick)

use anyhow::{Context, Result};
use bitvec::{order::Lsb0, vec::BitVec};
use clap::{Arg, Command};
use env_logger;
use newick::{one_from_filename, Newick, NodeID};
use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, Write},
};
use succ::{
    bitwise::{ops::NndOne, SparseOneNnd},
    tree::{
        balanced_parens::{BalancedParensTree, Node as BpNode},
        traversal::{DepthFirstTraverse, VisitNode},
        LabelVec, Node as SuccNode,
    },
};

// succ traversal from newick tree

type NwkTree = newick::NewickTree;

struct SuccTrav<'a> {
    t: &'a NwkTree,
    stack: Vec<(NodeID, usize, usize)>,
    lens: &'a mut Vec<f32>,                 // indexed by node-id
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
        for (k, &c) in self.t[id].children().iter().enumerate().rev() {
            self.stack.push((c, lvl + 1, k));
        }
        if self.lens.len() <= id {
            self.lens.resize(id + 1, 0.0);
        }
        self.lens[id] = self.t[id].branch().copied().unwrap_or(0.0);
        Some(VisitNode::new((), lvl, nth))
    }
}

// collect children in post-order

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

// for one pair only

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
        let a1 = in_a[v] == 1;
        let b1 = in_b[v] == 1;
        if a1 || b1 { union  += len; }
        if a1 && b1 { shared += len; }
    }
    if union == 0.0 { 0.0 } else { 1.0 - shared / union }
}

// read presence / absence table

fn read_table(p: &str) -> Result<(Vec<String>, Vec<String>, Vec<Vec<f64>>)> {
    let f = File::open(p)?;
    let mut lines = BufReader::new(f).lines();
    let hdr = lines.next().context("empty table")??;
    let mut it = hdr.split_whitespace(); it.next();
    let samples = it.map(|s| s.to_string()).collect();
    let mut taxa = Vec::new(); let mut mat = Vec::new();
    for l in lines {
        let row = l?;
        let mut p = row.split_whitespace();
        let tax = p.next().unwrap().to_string();
        let vals = p.map(|v| if v.parse::<f64>().unwrap_or(0.0) > 0.0 {1.0} else {0.0}).collect();
        taxa.push(tax); mat.push(vals);
    }
    Ok((taxa, samples, mat))
}

fn main() -> Result<()> {
    env_logger::Builder::from_default_env().init();
    let m = Command::new("unifrac_succ")
        .arg(Arg::new("tree").short('t').required(true))
        .arg(Arg::new("input").short('i').required(true))
        .arg(Arg::new("output").short('o').required(true))
        .get_matches();
    let tree_file = m.get_one::<String>("tree").unwrap();
    let tbl_file  = m.get_one::<String>("input").unwrap();
    let out_file  = m.get_one::<String>("output").unwrap();

    /* load tree and build BP */
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
            leaf_nm.push(t.name(n).cloned().unwrap_or_else(|| format!("L{n}")));
        }
    }

    /* allocate vectors: bp.len() + 1 covers virtual-root ID */
    let total = bp.len() + 1;
    lens.resize(total, 0.0);
    let mut kids = vec![Vec::<usize>::new(); total];
    let mut post = Vec::<usize>::with_capacity(total);
    collect_children::<SparseOneNnd>(&bp.root(), &mut kids, &mut post);

    /* table & bit-masks */
    let (taxa, samples, pres) = read_table(tbl_file)?;
    let nsamp = samples.len();
    let mut t2leaf = HashMap::<&str, usize>::new();
    for (i, n) in leaf_nm.iter().enumerate() { t2leaf.insert(n, i); }
    let mut masks: Vec<_> =
        (0..nsamp).map(|_| BitVec::repeat(false, leaf_ids.len())).collect();
    for (ti, tax) in taxa.iter().enumerate() {
        if let Some(&leaf) = t2leaf.get(tax.as_str()) {
            for (s, bits) in masks.iter_mut().enumerate() {
                if pres[ti][s] > 0.0 { bits.set(leaf, true); }
            }
        }
    }

    /* pairwise UniFrac */
    let mut dist = vec![0.0_f64; nsamp * nsamp];
    for i in 0..nsamp {
        for j in i+1..nsamp {
            let d = unifrac_pair(&post, &kids, &lens, &leaf_ids, &masks[i], &masks[j]);
            dist[i*nsamp + j] = d; dist[j*nsamp + i] = d;
        }
    }
    write_matrix(&samples, &dist, nsamp, out_file)?;
    Ok(())
}

/* ------------- write matrix ----------------- */

fn write_matrix(names: &[String], d: &[f64], n: usize, p: &str) -> Result<()> {
    let mut f = File::create(p)?;
    write!(f, "Sample")?;
    for s in names { write!(f, "\t{s}")?; } writeln!(f)?;
    for i in 0..n {
        write!(f, "{}", names[i])?;
        for j in 0..n { write!(f, "\t{:.6}", d[i*n + j])?; }
        writeln!(f)?;
    }
    Ok(())
}