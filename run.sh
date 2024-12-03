#!/bin/bash

set -e

bin/tf2stablehlo sample/frozen_graph.origin \
-o sample/frozen_graph.stablehlo \
--tf-input-arrays=X1,X2 \
--tf-input-data-types=DT_FLOAT,DT_FLOAT \
--tf-input-shapes=3,4:3,4 \
--tf-output-arrays=Sigmoid:0

find $(pwd)/sample/frozen_graph.{mlir,stablehlo}
echo

bin/compiler sample/frozen_graph.stablehlo
