import os
import random
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
import wandb
from peft import AdaLoraConfig, TaskType, get_peft_model
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
import time
import logging
import math

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load dataset
dataset = load_dataset("glue", "mrpc")

# Load tokenizer
model_name = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        padding="max_length",
        truncation=True,
        max_length=320,
    )


tokenized_datasets = dataset.map(tokenize_function, batched=True)


# Convert to PyTorch datasets with the necessary columns
def format_dataset(tokenized_datasets):
    return tokenized_datasets.map(
        lambda x: {
            "input_ids": x["input_ids"],
            "attention_mask": x["attention_mask"],
            "labels": x["label"],
        },
        remove_columns=["sentence1", "sentence2", "label", "idx", "token_type_ids"],
    )


train_dataset = format_dataset(tokenized_datasets["train"])
eval_dataset = format_dataset(tokenized_datasets["validation"])


# Define compute metrics function
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)

    precision = precision_score(
        labels, predictions, average="weighted", zero_division=0
    )
    recall = recall_score(labels, predictions, average="weighted", zero_division=0)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted", zero_division=0)

    return {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1,
    }


# Custom function to load model with size mismatch handling
def custom_load_state_dict(model, state_dict):
    model_dict = model.state_dict()
    for name, param in state_dict.items():
        if name in model_dict:
            if param.shape != model_dict[name].shape:
                print(f"Skipping {name} due to size mismatch")
                continue
        model_dict[name] = param
    model.load_state_dict(model_dict, strict=False)


# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# Function to initialize W&B with retry logic
def init_wandb(run_id):
    for attempt in range(3):  # Retry up to 3 times
        try:
            wandb.init(project="mrpc", name=f"adalora_run_{run_id}", reinit=True)
            return
        except Exception as e:
            logging.error(f"W&B initialization failed on attempt {attempt + 1}: {e}")
            time.sleep(5)  # Wait for 5 seconds before retrying
    raise RuntimeError("Failed to initialize W&B after 3 attempts")


# Run training and evaluation 5 times
for run in range(5):
    print(f"Run {run + 1}/5")

    # Set a different seed for each run
    seed_value = random.randint(1, 10000)
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=3000,
        load_best_model_at_end=False,
        metric_for_best_model="accuracy",  # Choose the metric for selecting the best model
        learning_rate=1e-3,
        per_device_train_batch_size=32,  # Set training batch size to 64
        per_device_eval_batch_size=32,  # Set evaluation batch size to 500
        num_train_epochs=30,
        weight_decay=0.01,
        warmup_ratio=0.1,
        seed=seed_value,
        logging_steps=100,
        report_to="wandb",
        logging_dir="./logs",
        fp16=True,
        dataloader_pin_memory=True,
        dataloader_drop_last=True,
    )

    # Reinitialize model for each run
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # LoRA Configuration
    adalora_config = AdaLoraConfig(
        target_r=8,
        init_r=10,
        beta1=0.85,
        beta2=0.85,
        tinit=600,
        tfinal=1800,
        deltaT=1,
        lora_alpha=32,
        lora_dropout=0.2,
        orth_reg_weight=0.1,
        total_step=math.ceil(
            len(train_dataset) / training_args.per_device_train_batch_size
        )
        * training_args.num_train_epochs,
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        target_modules=[
            "query_proj",
            "key_proj",
            "value_proj",  # Attention heads
            "intermediate.dense",
            "output.dense",
        ],
    )

    # Apply AdaLoRA
    model = get_peft_model(model, adalora_config)
    model.print_trainable_parameters()

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Initialize W&B run with retry logic
    try:
        init_wandb(run + 1)
    except RuntimeError as e:
        logging.error(f"Run {run + 1} failed to initialize W&B: {e}")
        continue  # Skip this run and move to the next iteration

    # Train and evaluate
    try:
        trainer.train()
    except Exception as e:
        logging.error(f"Run {run + 1} failed during training: {e}")
        wandb.finish()  # Ensure W&B run is properly finished
        continue  # Skip this run and move to the next iteration

    # Finish W&B run
    wandb.finish()

    # Clean up to prevent memory issues
    del model
    torch.cuda.empty_cache()

    # Add a delay between runs
    time.sleep(10)
