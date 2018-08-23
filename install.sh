#!/usr/bin/env bash
# make sure to run chmod +x install.sh

# first install Torch-7
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; echo y | bash install-deps
echo "yes" | ./install.sh
eval "$(cat ~/.bashrc | tail -n +10)"
echo "finished installing torch-7"

# get STN
echo "installing stn"
mkdir -p ~/torch/packages
echo "made dir"
cd ~/torch/packages
echo "in packages"
git clone https://github.com/qassemoquab/stnbhwd.git
cd stnbhwd
yes | luarocks make stnbhwd-scm-1.rockspec
echo "finished installing stn"

# get HDF5
echo "installing hdf5"
sudo apt-get -y install libhdf5-serial-dev
sudo apt-get -y install hdf5-tools
#this is for ubuntu 14.04...
if [ ! -d "/usr/include/hdf5/serial" ]; then
  sudo mkdir -p /usr/include/hdf5/serial
  sudo cp /usr/include/hdf5.h /usr/include/hdf5/serial/
  sudo cp /usr/include/hdf5_hl.h /usr/include/hdf5/serial/
fi
# continue with torch-hdf5 installation
cd ~/torch/packages
git clone https://github.com/deepmind/torch-hdf5
cd torch-hdf5
yes | luarocks make hdf5-0-0.rockspec LIBHDF5_LIBDIR="/usr/lib/x86_64-linux-gnu/"
echo "hdf5._config = {
    HDF5_INCLUDE_PATH = \"/usr/include/hdf5/serial\",
    HDF5_LIBRARIES = \"/usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5.so;/usr/lib/x86_64-linux-gnu/libpthread.so;/usr/lib/x86_64-linux-gnu/libsz.so;/usr/lib/x86_64-linux-gnu/libz.so;/usr/lib/x86_64-linux-gnu/libdl.so;/usr/lib/x86_64-linux-gnu/libm.so\"}">~/torch/install/share/lua/5.1/hdf5/config.lua
#
echo "finished installing hdf5"


# print tests
th -e "require 'hdf5'"
if [ $? -eq 0 ]; then
    echo -e "\e[32mhdf5 successfully installed\e[0m"
else
    echo -e "\e[31mhdf5 installation failed\e[0m"
fi

th -e "require 'stn'"
if [ $? -eq 0 ]; then
    echo -e "\e[32mstn successfully installed\e[0m"
else
    echo -e "\e[31mstn installation failed\e[0m"
fi
