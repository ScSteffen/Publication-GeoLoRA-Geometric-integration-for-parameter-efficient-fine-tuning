#!/bin/bash

# Define arrays for tau and lr values
tau_values="0.1" #"0.1 0.15 0.2 0.25 0.3)"  # Add more values if needed
lr_values="5e-4" #"1e-4 5e-4 1e-3 5e-3"   # Add more values if needed

# Loop over tau and lr values
for tau in $tau_values 
    do
    for lr in $lr_values 
        do
        # Execute the Python script with the current values of tau and lr
        python main_VIT_dlrt_lora.py --dataset_name cifar10 \
                                     --net_name vit_lora \
                                     --wandb 1 \
                                     --batch_size 48 \
                                     --epochs 20 \
                                     --lr $lr \
                                     --init_r 32 \
                                     --coeff_steps 0 \
                                     --tau $tau
    done
done
