#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/nvidia/current
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/tiger/cuda-10.0/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/tiger/cuda-10.0/nccl/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/tiger/cuda-10.0/extras/CUPTI/lib64

python create_toy_model.py