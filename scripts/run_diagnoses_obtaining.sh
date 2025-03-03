#!/bin/bash

export EXPERIMENT_NAME="llama_test"
export DEVICE="cuda:1"

python ../benchmarking/run_diagnoses_obtaining.py --experiment_name $EXPERIMENT_NAME \
    --device $DEVICE