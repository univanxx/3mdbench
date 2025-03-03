#!/bin/bash

export MODEL_NAME="your_model_name"
export COMPLAINTS_PATH="your_complaints_path"
export IMG_FOLDER="your_img_folder"
export DEVICE="your_device"
export EXPERIMENT_NAME="your_experiment_name"

python ../benchmarking/run_dialogue.py --model_name $MODEL_NAME \
    --complaints_path $COMPLAINTS_PATH --img_folder $IMG_FOLDER \
    --device $DEVICE --experiment_name $EXPERIMENT_NAME