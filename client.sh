#!/bin/bash

# Initialize Conda
source ~/miniconda3/etc/profile.d/conda.sh

# Define the base path
BASE_PATH="/home/niteshkumar/elexir/RankingResearch"

# Navigate to the specified directory
cd $BASE_PATH

# Activate the conda environment
conda activate llm

# Set the PYTHONPATH environment variable
export PYTHONPATH="${PYTHONPATH}:${BASE_PATH}"

python $BASE_PATH/llm_classifier/client_llm_classifier.py