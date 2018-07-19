#!/usr/bin/env bash
export PATH="$PATH:$HOME/bin"
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}} 
export LD_LIBRARY_PATH=/usr/local/cuda/lib64\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
python /home/djalab/Desktop/MARS_v1_7/MARS_v1_7.py 2>/home/djalab/Desktop/MARS_v1_7/MARS_v1_7.log
