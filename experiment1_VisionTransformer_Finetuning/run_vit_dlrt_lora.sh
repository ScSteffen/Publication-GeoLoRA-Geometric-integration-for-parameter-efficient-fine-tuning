#!/bin/bash

# Define arrays for tau and lr values
tau_values="0.15"  # Add more values if needed
lr_values="1e-3"   # Add more values if needed
run_id="1 2 3 4"
init_rs="2 4 8 16 32 64"

# Loop over tau and lr values
for run_id in $run_id
do
for init_r in $init_rs
do
for tau in $tau_values 
    do
    for lr in $lr_values 
        do
        # Execute the Python script with the current values of tau and lr
        python main_VIT_dlrt_lora.py --dataset_name cifar10 \
                                     --net_name vit_lora \
                                     --wandb 1 \
                                     --batch_size 256 \
                                     --epochs 5 \
                                     --lr $lr \
                                     --init_r $init_r \
                                     --coeff_steps 50 \
                                     --tau $tau
    done
done
done
done