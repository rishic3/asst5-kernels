#!/bin/bash

set -ex

MODE=${1:-"benchmark"}

# python wrap_cuda_submission.py rishic3
# cp templates/template.py submission.py

export PYTHONPATH="/opt/nvidia/nsight-compute/2025.1.1/extras/python:$PYTHONPATH"
export PYTHONPATH="$(dirname "$(pwd)"):$PYTHONPATH"

python ../eval.py $MODE test_cases/test.txt
