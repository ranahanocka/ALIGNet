# stn3dtorch
Torch Implementation of Volumetric Spatial Transformer Network which we used in [ALIGNet](https://github.com/ranahanocka/ALIGNet/). Code borrows heavily from the [PTN torch implementation](https://github.com/xcyan/ptnbhwd) and [2D STN torch implementation](https://github.com/qassemoquab/stnbhwd).

# Installation
```bash
git clone https://github.com/ranahanocka/stn3dtorch.git
cd stn3dtorch
luarocks make stn3dtorch-scm-1.rockspec
```

if everything went well, this import should work:
```bash
require 'volstn'
```
