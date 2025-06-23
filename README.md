# UniFrac implememtation in Rust

This is an example repo to show how to compute the [UniFrac](https://en.wikipedia.org/wiki/UniFrac) distance between a pair of samples containing taxa. 
It uses the succint data strucuture to represent a tree so that then the tree is huge, UniFrac computation can still be fast.

## Install
```bash
git clone https://github.com/jianshu93/unifrac_bp
cd unifrac_bp
cargo build --release
./target/release/unifrac_bp -h
```

## Usage 
```bash
Usage: unifrac -t <tree> -i <input> -o <output>

Options:
  -t <tree>        
  -i <input>       
  -o <output>      
  -h, --help       Print help
```

### example
```bash
### remove bootstrap support first if you have it

### Then run unifrac like this:
unifrac_bp -t data/test_rot_new2.nwk -i data/table.txt -o try.txt
cat try.txt
```

## References
1.Lozupone, C. and Knight, R., 2005. UniFrac: a new phylogenetic method for comparing microbial communities. Applied and environmental microbiology, 71(12), pp.8228-8235.

2.Hamady, M., Lozupone, C. and Knight, R., 2010. Fast UniFrac: facilitating high-throughput phylogenetic analyses of microbial communities including analysis of pyrosequencing and PhyloChip data. The ISME journal, 4(1), pp.17-27.

3.McDonald, D., Vázquez-Baeza, Y., Koslicki, D., McClelland, J., Reeve, N., Xu, Z., Gonzalez, A. and Knight, R., 2018. Striped UniFrac: enabling microbiome analysis at unprecedented scale. Nature methods, 15(11), pp.847-848.
