# UniFrac implememtation in Rust

This repo shows how to compute the [UniFrac](https://en.wikipedia.org/wiki/UniFrac) distance (both unweighted and weighted) between pairs of samples containing taxa. 
It uses the succint data strucuture (balanced parenthesis) in [succparen](https://github.com/sile/succparen.git) crate to represent a phylogenetic tree so that then the tree is huge, UniFrac computation can still be fast.

Striped UniFrac is the default algorithm and it is extremely fast for large number of samples. In fact, with sparse features of input samples, the complexity is close to O((N/s)^2), where s is average sparsity (average proportion of taxa detected at least once in pairs of samples/all taxa in the tree). An average sparsity of 5% indicates a 0.0025 scale down from O(N^2). 

Right now, the performance matches C++ version of Striped UniFrac in unifrac-binaries (https://github.com/biocore/unifrac-binaries) (CPU only) for ~4 thousand samples. I will stop optimizatizing here because this crate was developed for benchmark purposes.


## Install
```bash
git clone https://github.com/jianshu93/unifrac_bp
cd unifrac_bp

cargo build --release
./target/release/unifrac -h
```

## Usage 
```bash

 ************** initializing logger *****************

Striped UniFrac via Optimal Balanced Parenthesis

Usage: striped_unifrac [OPTIONS] --tree <tree> <--input <input>|--biom <biom>>

Options:
  -t, --tree <tree>        Input tree in Newick format
  -i, --input <input>      OTU/Feature table in TSV format
  -m, --biom <biom>        OTU/Feature table in BIOM (HDF5) format
      --weighted           Weighted UniFrac (normalized). Per-sample relative abundances will be used
  -T, --threads <threads>  Number of threads, default all logical cores
  -o, --output <output>    Output distance matrix in TSV format [default: unifrac.tsv]
  -h, --help               Print help
  -V, --version            Print version
```

### example
```bash
### Then run unifrac-rs like this:
./target/release/unifrac -t data/test.nwk -i data/test_OTU_table.txt  -o try.txt
cat try.txt

### Weighted 
./target/release/unifrac -t data/test.nwk -i data/test_OTU_table.txt --weighted  -o try.txt
```


### benchmark


```bash
#### Striped UniFrac, all logic cores/threads will be used by default
#### for 4204 microbiome samples with a total of ~0.5 million taxa, it took only ~30s on a M4 Max CPU. 
#### also ~30s on a Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz CPU with 32 cores
RUST_LOG=info ./target/release/unifrac -t ./GMTOLsong_table2024_N20_f2all_V4_table.nwk -m ./GMTOLsong_table2024_N20_f2all_V4_filt.biom -o GMTOLsong_dist_rs_biom.tsv
[2025-07-10T05:27:14Z INFO  unifrac_rs] Start parsing input.
[2025-07-10T05:27:14Z INFO  unifrac_rs] phase-1 masks built     40 ms
[2025-07-10T05:27:14Z INFO  unifrac_rs] phase-2 sparse lists built (66 strips)
[2025-07-10T05:27:45Z INFO  unifrac_rs] phase-3 block pass  31135 ms
[2025-07-10T05:27:46Z INFO  unifrac_rs] Start writing output.



### Weighted
$ RUST_LOG=info striped_unifrac -t ./GMTOLsong_table2024_N20_f2all_V4_table.nwk -m ./GMTOLsong_table2024_N20_f2all_V4_filt.biom --weighted -o GMTOLsong_dist_weighted_rs.tsv

 ************** initializing logger *****************

[2025-08-31T06:08:39Z INFO  striped_unifrac] logger initialized from default environment
[2025-08-31T06:08:42Z INFO  striped_unifrac] Total branches with positive length: 1039642
[2025-08-31T06:08:42Z INFO  striped_unifrac] Start parsing input.
[2025-08-31T06:08:53Z INFO  striped_unifrac] phase-1 (weighted)    245 ms
[2025-08-31T06:08:53Z INFO  striped_unifrac] phase-2 (weighted) lists built (62 strips)
[2025-08-31T06:09:24Z INFO  striped_unifrac] phase-3 (weighted)  12467 ms
[2025-08-31T06:09:24Z INFO  striped_unifrac] Start writing output.

```

For 50,000 samples on a 64-core AMD CPU. The performance matches that of C++ unifrac-binaries.
```bash
$unifrac -t ./ag_emp.tre -m ./ag_emp_even500.biom -o emp_50k_dist_rust.tsv

 ************** initializing logger *****************

real	12m17.507s
user	1486m0.383s
sys	3m17.306s


$time ssu -t ./ag_emp.tre -i ag_emp_even500.biom -o emp_50k_dist_c++.tsv -m unweighted

real
13m43.239s
user
108m10.510s
sys
0m50.950s

```

Weighted


```bash
$ RUST_LOG=info unifrac -t ./emp90.5000_1000_rxbl_placement_pruned75.tog.tre -m ./emp.90.min25.deblur.withtax.withtree.even1k.biom --weighted -o emp.90.weighted.tsv

 ************** initializing logger *****************

[2025-09-05T16:23:57Z INFO  striped_unifrac] logger initialized from default environment
[2025-09-05T16:24:03Z INFO  striped_unifrac] Total branches with positive length: 953222
[2025-09-05T16:24:03Z INFO  striped_unifrac] Start parsing input.
[2025-09-05T16:24:08Z INFO  striped_unifrac] parent pointers built in 4 ms
[2025-09-05T16:24:08Z INFO  striped_unifrac] block geometry: blk=64, nblk=393, threads=128
[2025-09-05T16:36:29Z INFO  striped_unifrac] weighted striped pass done in 740320 ms
[2025-09-05T16:36:29Z INFO  striped_unifrac] Start writing output.

```

### todo

GPU offloading via cudarc-rs

## References
1.Lozupone, C. and Knight, R., 2005. UniFrac: a new phylogenetic method for comparing microbial communities. Applied and environmental microbiology, 71(12), pp.8228-8235.

2.Lozupone, C.A., Hamady, M., Kelley, S.T. and Knight, R., 2007. Quantitative and qualitative β diversity measures lead to different insights into factors that structure microbial communities. Applied and environmental microbiology, 73(5), pp.1576-1585.

3.Hamady, M., Lozupone, C. and Knight, R., 2010. Fast UniFrac: facilitating high-throughput phylogenetic analyses of microbial communities including analysis of pyrosequencing and PhyloChip data. The ISME journal, 4(1), pp.17-27.

4.McDonald, D., Vázquez-Baeza, Y., Koslicki, D., McClelland, J., Reeve, N., Xu, Z., Gonzalez, A. and Knight, R., 2018. Striped UniFrac: enabling microbiome analysis at unprecedented scale. Nature methods, 15(11), pp.847-848.

5.Sfiligoi, I., Armstrong, G., Gonzalez, A., McDonald, D. and Knight, R., 2022. Optimizing UniFrac with OpenACC yields greater than one thousand times speed increase. Msystems, 7(3), pp.e00028-22.
