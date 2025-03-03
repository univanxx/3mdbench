#!/bin/bash

export EXPERIMENT_NAME="your_experiment_name"
export COMPLAINTS_PATH="your_complaints_path"
export IMG_FOLDER="your_img_folder"
export DEVICE="your_device"

python ../benchmarking/run_assessment.py --experiment_name $EXPERIMENT_NAME \
    --complaints_path $COMPLAINTS_PATH --img_folder $IMG_FOLDER \
    --device $DEVICE