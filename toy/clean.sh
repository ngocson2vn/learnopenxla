#!/bin/bash

# cd toy_model_v1/dump
# rm -rfv cache_dir/ cluster.mlir fuse.info kernel.* libkernel.so metric.json opt_score predict_online/ tf_to_kernel.std* function.pb
# cd -

rm -rvf toy_model_debug
echo

mkdir toy_model_debug
cp -v origin/* toy_model_debug/
