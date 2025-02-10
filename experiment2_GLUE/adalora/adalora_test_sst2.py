import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
from peft import AdaLoraConfig, get_peft_model
import wandb
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import logging
import os
import random


# Configure logging
logging.basicConfig(level=logging.INFO)

# Load dataset
dataset = load_dataset("glue", "sst2")

# Load tokenizer
model_name = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length = 128) #https://github.com/QingruZhang/AdaLoRA/blob/d10f5ebee16c478fa2f41a44a237b38e8c9b0338/NLU/scripts/run_debertav3_sst2.sh#L13C3-L13C21


tokenized_datasets = dataset.map(tokenize_function, batched=True)
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["validation"]

# Define compute metrics function
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)

    precision = precision_score(labels, predictions, average="weighted")
    recall = recall_score(labels, predictions, average="weighted")
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")

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
            wandb.init(project="sst2", name=f"adalora_run_{run_id}", reinit=True)
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
        eval_steps=3000,
        save_steps=9000,
        per_device_train_batch_size=32,  # Set training batch size to 64
        per_device_eval_batch_size=32,  # Set evaluation batch size to 500
        load_best_model_at_end=False,
        learning_rate=8e-4,               #https://github.com/QingruZhang/AdaLoRA/blob/d10f5ebee16c478fa2f41a44a237b38e8c9b0338/NLU/scripts/run_debertav3_sst2.sh#L15
        #per_device_train_batch_size=32,   #https://github.com/QingruZhang/AdaLoRA/blob/d10f5ebee16c478fa2f41a44a237b38e8c9b0338/NLU/scripts/run_debertav3_sst2.sh#L14
        num_train_epochs=24,              #https://github.com/QingruZhang/AdaLoRA/blob/d10f5ebee16c478fa2f41a44a237b38e8c9b0338/NLU/scripts/run_debertav3_sst2.sh#L16
        weight_decay=0.01,                #https://github.com/QingruZhang/AdaLoRA/blob/d10f5ebee16c478fa2f41a44a237b38e8c9b0338/NLU/scripts/run_debertav3_sst2.sh#L17
        warmup_steps=1000,                #https://github.com/QingruZhang/AdaLoRA/blob/d10f5ebee16c478fa2f41a44a237b38e8c9b0338/NLU/scripts/run_debertav3_sst2.sh#L17
        seed=seed_value,
        logging_steps=100,
        report_to="wandb", 
        logging_dir="./logs"
    )

    # Reinitialize model for each run
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # LoRA Configuration
    adalora_config = AdaLoraConfig(
        target_r=8,
        init_r=10,
        tinit=6000,
        tfinal=22000,
        deltaT=100,
        beta1=0.85,
        beta2=0.85,
        #orth_reg_weight=0.3,
        target_modules=[
            "query_proj", "key_proj", "value_proj",
            "intermediate.dense", "output.dense",
            "classifier"
        ],
        lora_alpha=16,
        lora_dropout=0.2,
        task_type="SEQ_CLS"
    )

    # Apply LoRA
    model = get_peft_model(model, adalora_config)

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

    # Save the best model
    best_model_checkpoint = trainer.state.best_model_checkpoint
    if best_model_checkpoint is not None and os.path.exists(f"{best_model_checkpoint}/pytorch_model.bin"):
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        model = get_peft_model(model, adalora_config)
        state_dict = torch.load(f"{best_model_checkpoint}/pytorch_model.bin")
        custom_load_state_dict(model, state_dict)
    else:
        logging.warning(f"Best model checkpoint not found for run {run + 1}")

    # Clean up to prevent memory issues
    del model
    torch.cuda.empty_cache()

    # Add a delay between runs
    time.sleep(10)
