# UniFrac implememtation in Rust

This repo shows how to compute the [UniFrac](https://en.wikipedia.org/wiki/UniFrac) distance between pairs of samples containing taxa. 
It uses the succint data strucuture (balanced parenthesis) in [succparen](https://github.com/sile/succparen.git) crate to represent a phylogenetic tree so that then the tree is huge, UniFrac computation can still be fast.

Striped UniFrac is the default algorithm and it is extremely fast for large number of samples. In fact, with sparse features of input samples, the complexity is close to O((N/s)^2), where s is average sparsity (average proportion of taxa detected at least once in pairs of samples/all taxa in the tree). An average sparsity of 5% indicates a 0.0025 scale down from O(N^2). 

Right now, the performance matches C++ version of Striped UniFrac in unifrac-binaries (https://github.com/biocore/unifrac-binaries) (CPU only) for ~4 thousand samples. I will stop optimizatizing here because this crate was developed for metagenomic UniFrac and non-linear UniFrac embedding and we never reach such large scale for metagenomic sampling. 


## Install
```bash
git clone https://github.com/jianshu93/unifrac_bp
cd unifrac_bp

#### you must have hdf5 installed and library in the path, cmake and gcc is also required for static compiling of hdf5
cargo build --release
./target/release/unifrac -h
```

## Usage 
```bash

 ************** initializing logger *****************

Striped UniFrac via Optimal Balanced Parenthesis

Usage: unifrac [OPTIONS] --tree <tree> <--input <input>|--biom <biom>>

Options:
  -t, --tree <tree>      Input tree in Newick format
  -i, --input <input>    OTU/Feature table in TSV format
  -m, --biom <biom>      OTU/Feature table in BIOM (HDF5) format
  -o, --output <output>  Output distance matrix in TSV format [default: unifrac.tsv]
  -h, --help             Print help
  -V, --version          Print version
```

### example
```bash
### Then run unifrac-rs like this:
./target/release/unifrac -t data/test.nwk -i data/test_OTU_table.txt  -o try.txt
cat try.txt
```


### benchmark


```bash
#### Striped UniFrac, all logic cores/threads will be used by default
#### for 4204 microbiome samples with a total of ~0.5 million taxa, it took only ~30s on a M4 Max CPU. 
#### also ~30s on a Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz CPU with 32 cores
RUST_LOG=info ./target/release/unifrac -t ./GMTOLsong_table2024_N20_f2all_V4_table.nwk -m ./GMTOLsong_table2024_N20_f2all_V4_table.biom -o GMTOLsong_dist_rs_biom.tsv
[2025-07-10T05:27:14Z INFO  unifrac_rs] Start parsing input.
[2025-07-10T05:27:14Z INFO  unifrac_rs] phase-1 masks built     40 ms
[2025-07-10T05:27:14Z INFO  unifrac_rs] phase-2 sparse lists built (66 strips)
[2025-07-10T05:27:45Z INFO  unifrac_rs] phase-3 block pass  31135 ms
[2025-07-10T05:27:46Z INFO  unifrac_rs] Start writing output.
```

### todo

1.Weighted UniFrac

2.SIMD (another 2-5 times speedup for the block pass phase)

3.GPU offloading

## References
1.Lozupone, C. and Knight, R., 2005. UniFrac: a new phylogenetic method for comparing microbial communities. Applied and environmental microbiology, 71(12), pp.8228-8235.

2.Hamady, M., Lozupone, C. and Knight, R., 2010. Fast UniFrac: facilitating high-throughput phylogenetic analyses of microbial communities including analysis of pyrosequencing and PhyloChip data. The ISME journal, 4(1), pp.17-27.

3.McDonald, D., Vázquez-Baeza, Y., Koslicki, D., McClelland, J., Reeve, N., Xu, Z., Gonzalez, A. and Knight, R., 2018. Striped UniFrac: enabling microbiome analysis at unprecedented scale. Nature methods, 15(11), pp.847-848.

4.Sfiligoi, I., Armstrong, G., Gonzalez, A., McDonald, D. and Knight, R., 2022. Optimizing UniFrac with OpenACC yields greater than one thousand times speed increase. Msystems, 7(3), pp.e00028-22.
