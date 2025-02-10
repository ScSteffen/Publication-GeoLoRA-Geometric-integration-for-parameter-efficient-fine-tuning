import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import wandb
import numpy as np
from scipy.stats import pearsonr, spearmanr
import time
import logging
import os
import random


# Configure logging
logging.basicConfig(level=logging.INFO)

# Load dataset
dataset = load_dataset("glue", "stsb")

# Load tokenizer
model_name = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True, max_length=128)   #https://github.com/QingruZhang/AdaLoRA/blob/d10f5ebee16c478fa2f41a44a237b38e8c9b0338/NLU/scripts/run_debertav3_stsb.sh#L12

tokenized_datasets = dataset.map(tokenize_function, batched=True)
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["validation"]

# Ensure datasets have length and proper format
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Define compute metrics function
def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.flatten()  # Ensure predictions is 1-dimensional

    pearson_corr = pearsonr(predictions, labels)[0]
    spearman_corr = spearmanr(predictions, labels)[0]

    return {
        "pearson": pearson_corr,
        "spearman": spearman_corr,
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
            wandb.init(project="stsb", name=f"run_{run_id}", reinit=True)
            return
        except Exception as e:
            logging.error(f"W&B initialization failed on attempt {attempt + 1}: {e}")
            time.sleep(5)  # Wait for 5 seconds before retrying
    raise RuntimeError("Failed to initialize W&B after 3 attempts")

# Run training and evaluation 5 times
for run in range(5):
    print(f"Run {run + 1}/5")

    # Set a different random seed for each run
    seed_value = random.randint(1, 10000)
    
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=10000,
        per_device_train_batch_size=32,  # Set training batch size to 64
        per_device_eval_batch_size=32,  # Set evaluation batch size to 500
        load_best_model_at_end=False,
        learning_rate=2.2e-5,            #https://github.com/QingruZhang/AdaLoRA/blob/d10f5ebee16c478fa2f41a44a237b38e8c9b0338/NLU/scripts/run_debertav3_stsb.sh#L13
        #per_device_train_batch_size=32,  #https://github.com/QingruZhang/AdaLoRA/blob/d10f5ebee16c478fa2f41a44a237b38e8c9b0338/NLU/scripts/run_debertav3_stsb.sh#L13
        num_train_epochs=25,             #https://github.com/QingruZhang/AdaLoRA/blob/d10f5ebee16c478fa2f41a44a237b38e8c9b0338/NLU/scripts/run_debertav3_stsb.sh#L14
        weight_decay=0.1,                #https://github.com/QingruZhang/AdaLoRA/blob/d10f5ebee16c478fa2f41a44a237b38e8c9b0338/NLU/scripts/run_debertav3_stsb.sh#L15
        warmup_steps=100,                #https://github.com/QingruZhang/AdaLoRA/blob/d10f5ebee16c478fa2f41a44a237b38e8c9b0338/NLU/scripts/run_debertav3_stsb.sh#L14
        seed=seed_value,
        logging_steps=100,
        report_to="wandb",
        logging_dir="./logs",
        fp16=True,
        dataloader_pin_memory=True,
        dataloader_drop_last=True,
       # gradient_accumulation_steps=1,  # Start with 1 and adjust as needed
       # gradient_checkpointing=False,
    )

    # Reinitialize model for each run
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    # LoRA Configuration without the additional parameters
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,           #https://github.com/QingruZhang/AdaLoRA/blob/d10f5ebee16c478fa2f41a44a237b38e8c9b0338/NLU/scripts/run_debertav3_stsb.sh#L11
        target_modules=[         #https://github.com/QingruZhang/AdaLoRA/blob/d10f5ebee16c478fa2f41a44a237b38e8c9b0338/NLU/scripts/run_debertav3_stsb.sh#L10
            "query_proj", "key_proj", "value_proj",
            "intermediate.dense", "output.dense",
            "classifier"
        ],
        lora_dropout=0.2,
        bias="none",
        task_type="SEQ_CLS"
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset.shuffle(seed=seed_value),  # Shuffle with the same random seed
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Initialize W&B run with retry logic
    try:
        init_wandb(run + 1)
    except RuntimeError as e:
        logging.error(f"Run {run + 1} failed to initialize W&B: {e}")
        continue

    # Train and evaluate
    try:
        trainer.train()
    except Exception as e:
        logging.error(f"Run {run + 1} failed during training: {e}")
        wandb.finish()
        continue
    
    # Finish W&B run
    wandb.finish()

    # Save the best model
    best_model_checkpoint = trainer.state.best_model_checkpoint
    if best_model_checkpoint is not None and os.path.exists(f"{best_model_checkpoint}/pytorch_model.bin"):
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
        model = get_peft_model(model, lora_config)
        state_dict = torch.load(f"{best_model_checkpoint}/pytorch_model.bin")
        custom_load_state_dict(model, state_dict)
    else:
        logging.warning(f"Best model checkpoint not found for run {run + 1}")

    # Clean up to prevent memory issues
    del model
    torch.cuda.empty_cache()

    # Add a delay between runs
    time.sleep(10)
