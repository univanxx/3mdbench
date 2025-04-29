#!/bin/bash

export MODEL_NAME="your_model_name"
export DEVICE="your_device"
export EXPERIMENT_NAME="your_experiment_name"

python ../benchmarking/run_dialogue.py --model_name $MODEL_NAME \
                                       --device $DEVICE \
                                       --experiment_name $EXPERIMENT_NAME