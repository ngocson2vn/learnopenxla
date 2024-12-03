#!/bin/bash

set -e

###################################################################################################
# Prerequisites
###################################################################################################
# python
# pyenv install 3.9.0
# pyenv global 3.9.0

python_version=$(python -V | awk '{print $2}')
echo "Python version: ${python_version}"
if ! echo ${python_version} | grep '3\.9' >/dev/null; then
  echo "ERROR: Python version is lower than 3.9.x!"
  exit 1
fi

# gcc --version
# gcc (GCC) 10.1.0
gcc_version=$(env gcc --version | head -1 | awk '{print $3}')
echo "GCC version: ${gcc_version}"
if ! echo ${gcc_version} | grep '10\.' >/dev/null; then
  echo "ERROR: GCC version is lower than 10.x!"
  exit 1
fi

export TF_PYTHON_VERSION=3.9
touch requirements_lock_3_9.txt

###################################################################################################
# Initialization
###################################################################################################
if [ ! -f ./bazel ]; then
  echo "[bazel install]"
  wget https://github.com/bazelbuild/bazelisk/releases/download/v1.20.0/bazelisk-linux-amd64
  chmod +x bazelisk-linux-amd64
  ln -sf bazelisk-linux-amd64 bazel
else
  echo "bazel has been installed already"
  ./bazel --version
  echo
fi

# git config -f .gitmodules submodule.third_party/tensorflow.shallow true
# git submodule update --init --recursive

echo "tensorflow submodule"
git submodule status
echo

# Patch tensorflow
pushd third_party/tensorflow/
if [ ! -f patch.done ]; then
  echo "START patching tensorflow"
  git apply ../../patches/tensorflow/*.patch
  touch patch.done
  echo "DONE patching tensorflow"
fi
popd

yes | cp -vf third_party/tensorflow/.bazelversion .
yes | cp -vf third_party/tensorflow/.bazelrc .

###################################################################################################
# Build targets
###################################################################################################
export TF_NEED_ROCM=0
export TF_NEED_OPENCL_SYCL=0
export TF_NEED_CUDA=1
export TF_NEED_TENSORRT=0
export TF_CUDA_PATHS="/usr/local/cuda-12.2"
export TF_CUDA_VERSION="12.2"
export TF_CUDNN_VERSION="8.9"
export CUDA_TOOLKIT_PATH="/usr/local/cuda-12.2"
export TF_CUDA_COMPUTE_CAPABILITIES="6.1,7.0,7.5,8.0,8.6,9.0"
export TF_CUDA_CLANG="0"
export GCC_HOST_COMPILER_PATH="/usr/bin/gcc"

rm -rf output
mkdir -p output/lib

FLAGS="--define=no_nccl_support=true --define=no_aws_support=true --define=no_gcp_support=true --define=no_hdfs_support=true"

# COPTS="-O0,-g,-fno-inline"
# SRC_FILES=+tf2stablehlo.cc
# SRC_FILES=${SRC_FILES},+tensorflow/compiler/mlir/quantization/tensorflow/quantize_preprocess.cc
# SRC_FILES=${SRC_FILES},+tensorflow/compiler/mlir/tensorflow/transforms/shape_inference_pass.cc
# SRC_FILES=${SRC_FILES},+tensorflow/compiler/mlir/tensorflow/transforms/shape_inference.cc
# CC=/usr/bin/gcc ./bazel --output_user_root=./forge build //:tf2stablehlo \
# --per_file_copt=${SRC_FILES}@${COPTS} \
# --strip=never $FLAGS \
# --experimental_repo_remote_exec \
# --verbose_failures \
# --sandbox_debug \
# -j 128

# cp -v  bazel-bin/tf2stablehlo output/
# cp -Lv bazel-bin/external/org_tensorflow/tensorflow/libtensorflow_framework.so.2 output/lib/
# pushd output
# chmod +w tf2stablehlo
# patchelf --force-rpath --set-rpath ./lib tf2stablehlo
# popd
# cp -rv sample output/
# cp -v  run.sh output/

COPTS="-O0,-g,-fno-inline"
SRC_FILES=+stablehlo_compiler.cc
CC=/usr/bin/gcc ./bazel --output_user_root=./forge build --config=cuda -s //:stablehlo_compiler \
--per_file_copt=${SRC_FILES}@${COPTS} \
--strip=never $FLAGS \
--experimental_repo_remote_exec \
--verbose_failures \
--sandbox_debug \
--experimental_ui_max_stdouterr_bytes=10485760
-j `nproc`

cp -v  bazel-bin/stablehlo_compiler output/
pushd output
chmod +w stablehlo_compiler
patchelf --force-rpath --set-rpath ./lib stablehlo_compiler
popd
cp -v  compile.sh output/
