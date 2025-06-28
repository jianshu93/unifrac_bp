# UniFrac implememtation in Rust

This repo is designed to compute the [UniFrac](https://en.wikipedia.org/wiki/UniFrac) distance between pairs of samples containing taxa. 
It uses the succint data strucuture (balanced parenthesis) to represent a phylogenetic tree so that then the tree is huge, UniFrac computation can still be fast.

Striped UniFrac can also be used via the --striped option to be extremely fast for large number of samples. In fact, with sparse features of input samples, the complexity is close to O((N/s)^2), where s is average sparsity (average proportion of taxa detected at least once in pairs of samples/all taxa in the tree). An average sparsity of 5% indicates a 0.0025 scale down from O(N^2). 

Right now, the performance matches C++ version of Striped UniFrac in unifrac-binaries (https://github.com/biocore/unifrac-binaries) for ~20 thousand samples (the largest sample collection we have till today).


## Install
```bash
git clone https://github.com/jianshu93/unifrac_bp
cd unifrac_bp
cargo build --release
./target/release/unifrac_bp -h
```

## Usage 
```bash
Usage: unifrac [OPTIONS] --tree <tree> --input <input>

Options:
  -t, --tree <tree>      Input tree in Newick format
  -i, --input <input>    Input OTU table in TSV format
  -o, --output <output>  Output distance matrix in TSV format [default: unifrac.tsv]
      --striped          Use striped UniFrac algorithm
  -h, --help             Print help
```

### example
```bash
### remove bootstrap support first if you have it

### Then run unifrac like this:
unifrac -t data/test.nwk -i data/test_OTU_table.txt  -o try.txt
cat try.txt
```

## References
1.Lozupone, C. and Knight, R., 2005. UniFrac: a new phylogenetic method for comparing microbial communities. Applied and environmental microbiology, 71(12), pp.8228-8235.

2.Hamady, M., Lozupone, C. and Knight, R., 2010. Fast UniFrac: facilitating high-throughput phylogenetic analyses of microbial communities including analysis of pyrosequencing and PhyloChip data. The ISME journal, 4(1), pp.17-27.

3.McDonald, D., Vázquez-Baeza, Y., Koslicki, D., McClelland, J., Reeve, N., Xu, Z., Gonzalez, A. and Knight, R., 2018. Striped UniFrac: enabling microbiome analysis at unprecedented scale. Nature methods, 15(11), pp.847-848.
