#!/bin/bash

set -e

./bazel clean --expunge
rm -rf ./forge