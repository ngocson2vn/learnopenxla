#!/bin/bash

cwd=$(pwd)
base_dir=$(basename $(dirname $cwd))
workspace_dir=$(dirname $(dirname $cwd))
echo workspace_dir=${workspace_dir}

MODEL_NAME=toy_model_v1

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/nvidia/current
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/tiger/cuda-10.0/lib64
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/tiger/cuda-10.0/nccl/lib
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/tiger/cuda-10.0/extras/CUPTI/lib64

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.4/lib64/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.4/nccl/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.4/extras/CUPTI/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu


export PYTHONPATH=/opt/tiger:/opt/tiger/pyutil
export AUTO_FUSION_REGION=VA
export ERDOS_OPS_BYTED_CIFACE_PATH=/opt/tiger/auto_fusion_sdk/lib/libbyted_ciface_dyn_py_1_15_5.so
export ERDOS_OPS_KERNEL_PATH=${MODEL_NAME}/optimize/libkernel.so

echo "ERDOS_OPS_BYTED_CIFACE_PATH: ${ERDOS_OPS_BYTED_CIFACE_PATH}"
echo "ERDOS_OPS_KERNEL_PATH: ${ERDOS_OPS_KERNEL_PATH}"

# strace python run_toy_model.py -m toy_model_debug
python run_toy_model.py -m ${MODEL_NAME}