#!/bin/bash

# Define arrays for tau and lr values
tau_values="100 200 300 400 500 600"  # Add more values if needed
lr_values="1e-1 1e-2 1e-3 1e-4"   # Add more values if needed
run_id="1 2 3"

# Loop over tau and lr values
for run_id in $run_id
do
for tau in $tau_values 
    do
    for lr in $lr_values 
        do
        # Execute the Python script with the current values of tau and lr
        python main_VIT_baselora.py --dataset_name cifar100 \
                                    --net_name vit_adalora \
                                    --wandb 1 \
                                    --batch_size 256 \
                                    --epochs 5 \
                                    --lr $lr \
                                    --init_r 16 \
                                    --coeff_steps 50 \
                                    --rank_budget $tau
        done
done
done