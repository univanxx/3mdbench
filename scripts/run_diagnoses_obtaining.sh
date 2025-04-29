#!/bin/bash

export EXPERIMENT_NAME="your_experiment_name"
export DEVICE="your_device"

python ../benchmarking/run_diagnoses_obtaining.py --experiment_name $EXPERIMENT_NAME \
                                                  --device $DEVICE