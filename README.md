# ALIGNet <img src="https://dl.dropboxusercontent.com/s/anyszxm1phvdjfh/orig.png?dl=0" width="50px"/> in Torch

[[Project]](https://ranahanocka.github.io/ALIGNet/)   [[Arxiv]](https://bit.ly/alignet) <br>
ALIGNet is a network trained to register pairs of shapes using a learned data-driven prior, and doesn't need ground-truth warp fields for supervision. 

### Key Idea
If I asked you to deform the blue "H" to the the orange "H",  you will be able to perform the deformation even where the "H" has a missing piece due to your prior knowledge about the letter H

<img src="https://www.dropbox.com/s/4o4omqoca5snrp9/output.gif?raw=1" width="450px"/> 

ALIGNet learns a data-driven prior which guides the alignment both in the missing and complete regions of the shape.
Some results:

<img src="docs/rep.png" width="450px"/> 

where the pink region is a visualization of the missing piece in the target shape. The estimated alignments are oblivious to missing pieces in the partial shape. 

The code was written by [Rana Hanocka](https://www.cs.tau.ac.il/~hanocka/) with support from [Noa Fish](http://www.cs.tau.ac.il/~noafish/) and [Zhenhua Wang](http://zhwang.me).

*This repo is still under active development*
# Setup
### Prerequisites
- Linux (tested on Ubuntu 16.04, 14.04 and Linux Mint)
- NVIDIA GPU + CUDA (tested on cuda8 and cuda7.5) *should also work on CPU*

### Getting Started
- Clone this repo:
```bash
git clone https://github.com/ranahanocka/ALIGNet.git
cd ALIGNet
```
- Install all dependencies: [Torch-7](http://torch.ch/docs/getting-started.html), [STN](https://github.com/qassemoquab/stnbhwd), [HDF5](https://github.com/deepmind/torch-hdf5):
```bash
sudo ls
chmod +x install.sh
./install.sh
```


# Training
- download the data
```bash
chmod +x download_data.sh
./download_data.sh
```
- run training
```bash
th main.lua -data /path/to/data
```


# Citation
If you find this code useful, please consider citing our paper
```
@article{hanocka2018alignet,
 author = {Hanocka, Rana and Fish, Noa and Wang, Zhenhua and Giryes, Raja and Fleishman, Shachar and Cohen-Or, Daniel},
 title = {ALIGNet: Partial-Shape Agnostic Alignment via Unsupervised Learning},
 journal = {ACM Trans. Graph.},
 year = {2018}}

```

# Contributing
Contributions to this repository are very welcome. Open an issue if you: have problems running the code, want to suggest improvements, or want to submit a pull request.

# Acknowledgments
The code design and multithreading data loading capabilities used in this code were adopted from [imagenet-multiGPU.torch](https://github.com/soumith/imagenet-multiGPU.torch).
