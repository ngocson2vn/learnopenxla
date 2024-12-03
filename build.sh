#!/bin/bash

set -e

###################################################################################################
# Prerequisites
###################################################################################################
# python
# pyenv install 3.11.0
# pyenv global 3.11.0

python_version=$(python -V | awk '{print $2}')
echo "Python version: ${python_version}"
if ! echo ${python_version} | grep '3\.11' >/dev/null; then
  echo "ERROR: Python version is lower than 3.11.x!"
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

export TF_PYTHON_VERSION=3.11
# touch requirements_lock_3_9.txt
touch requirements_lock_3_11.txt

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

git submodule update --init --recursive

echo "git submodule status"
git submodule status
echo


###################################################################################################
# Build tf2stablehlo
###################################################################################################
echo "============================="
echo "Build tf2stablehlo"
echo "============================="

yes | cp -vf tf2stablehlo/workspace.bzl ./WORKSPACE

# Patch tensorflow
pushd tf2stablehlo/third_party/tensorflow/
if [ ! -f patch.done ]; then
  echo "START patching tensorflow"
  git apply ../../patches/tensorflow/*.patch
  touch patch.done
  echo "DONE patching tensorflow"
fi
popd

yes | cp -vf tf2stablehlo/third_party/tensorflow/.bazelversion .
yes | cp -vf tf2stablehlo/third_party/tensorflow/.bazelrc .

export TF_NEED_ROCM=0
export TF_NEED_OPENCL_SYCL=0
export TF_NEED_CUDA=1
export TF_NEED_TENSORRT=0
export TF_CUDA_VERSION="12.2"
export HERMETIC_CUDA_VERSION="12.2.0"
export TF_CUDNN_VERSION="8.9"
export HERMETIC_CUDNN_VERSION="8.9.7.29"
export TF_CUDA_PATHS="/usr/local/cuda-12.2"
export CUDA_TOOLKIT_PATH="/usr/local/cuda-12.2"
export TF_CUDA_COMPUTE_CAPABILITIES="6.1,7.0,7.5,8.0,8.6,9.0"
export TF_CUDA_CLANG="0"
export GCC_HOST_COMPILER_PATH="/usr/bin/gcc"

rm -rf output
mkdir -p output/bin
mkdir -p output/lib

FLAGS="--define=no_nccl_support=true --define=no_aws_support=true --define=no_gcp_support=true --define=no_hdfs_support=true"

COPTS="-O0,-g,-fno-inline"
SRC_FILES=+tf2stablehlo.cc
SRC_FILES=${SRC_FILES},+tensorflow/compiler/mlir/quantization/tensorflow/quantize_preprocess.cc
SRC_FILES=${SRC_FILES},+tensorflow/compiler/mlir/tensorflow/transforms/shape_inference_pass.cc
SRC_FILES=${SRC_FILES},+tensorflow/compiler/mlir/tensorflow/transforms/shape_inference.cc
CC=/usr/bin/gcc ./bazel --output_user_root=tf2stablehlo/build build //tf2stablehlo:tf2stablehlo \
--per_file_copt=${SRC_FILES}@${COPTS} \
--strip=never $FLAGS \
--experimental_repo_remote_exec \
--verbose_failures \
--sandbox_debug \
-j `nproc`

cp -v  bazel-bin/tf2stablehlo/tf2stablehlo output/bin
cp -Lv bazel-bin/external/org_tensorflow/tensorflow/libtensorflow_framework.so.2 output/lib/
pushd output
chmod +w bin/tf2stablehlo
patchelf --force-rpath --set-rpath ./lib bin/tf2stablehlo
popd
cp -rv sample output/
cp -v  run.sh output/
echo && echo

###################################################################################################
# Build compiler
###################################################################################################
echo "============================="
echo "Build compiler"
echo "============================="

yes | cp -vf compiler/workspace.bzl ./WORKSPACE

yes | cp -vf compiler/third_party/xla/.bazelversion .
yes | cp -vf compiler/third_party/xla/.bazelrc .

COPTS="-O0,-g,-fno-inline"
SRC_FILES=+compiler.cc
CC=/usr/bin/gcc ./bazel --output_user_root=compiler/build build --config=cuda --cxxopt=-v //compiler:compiler \
--repo_env=HERMETIC_CUDA_VERSION="12.2.0" \
--repo_env=HERMETIC_CUDNN_VERSION="8.9.7.29" \
--per_file_copt=${SRC_FILES}@${COPTS} \
--strip=never $FLAGS \
--verbose_failures \
--sandbox_debug \
--experimental_ui_max_stdouterr_bytes=10485760 \
-j `nproc`

cp -v bazel-bin/compiler/compiler output/bin
pushd output
chmod +w bin/compiler
patchelf --force-rpath --set-rpath ./lib bin/compiler
popd
