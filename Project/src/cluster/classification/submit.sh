#!/usr/bin/env bash

# Path to the `.py` file you want to run
PYTHON_SCRIPT_PATH="/home/hlcv_team015/hlcv/Project/src/common/"
# Path to the Python binary of the conda environment
CONDA_PYTHON_BINARY_PATH="/home/hlcv_team015/miniconda3/envs/hlcv/bin/python"

cd $PYTHON_SCRIPT_PATH
$CONDA_PYTHON_BINARY_PATH "$@"