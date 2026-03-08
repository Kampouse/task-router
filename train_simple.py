#!/usr/bin/env python3
"""Simple training script without Optuna (since first trial hit 100% accuracy)."""

import json
import numpy as np
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
import torch

# Task labels
TASK_LABELS = ["coding", "cooking", "creative", "embedding", "general", "math_reasoning", "qa", "research", "summarization", "tools", "translation", "vision"]
label2id = {label: i for i, label in enumerate(TASK_LABELS)}
id2label = {i: label for label, i in label2id.items()}

# Tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")


def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )


def encode_labels(batch):
    batch["label"] = label2id[batch["label"]]
    return batch


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Task Router Training")
    print("=" * 60)
    print(f"Labels: {TASK_LABELS}")
    print(f"Device: {'mps' if torch.backends.mps.is_available() else 'cpu'}")
    print()
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("json", data_files="tasks.jsonl", split="train")
    
    # Preprocess
    print("Preprocessing...")
    dataset = dataset.map(encode_labels)
    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    # Split
    splits = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = splits["train"]
    eval_dataset = splits["test"]
    
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    
    # Model
    print("Loading model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(TASK_LABELS),
        id2label=id2label,
        label2id=label2id,
    )
    
    # Training arguments (optimal params from first Optuna trial)
    training_args = TrainingArguments(
        output_dir="./task-router",
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2.5e-4,
        weight_decay=0.06,
        warmup_ratio=0.18,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=10,
        report_to="none",
        seed=42,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("\nTraining...")
    trainer.train()
    
    # Evaluate
    print("\nEvaluating...")
    result = trainer.evaluate()
    print(f"Accuracy: {result['eval_accuracy']:.2%}")
    print(f"F1 Score: {result['eval_f1']:.2%}")
    
    # Save
    print("\nSaving model...")
    trainer.save_model("./task-router")
    tokenizer.save_pretrained("./task-router")
    
    # Save config
    with open("./task-router/config.json", "r+") as f:
        config = json.load(f)
        config["task_labels"] = TASK_LABELS
        f.seek(0)
        json.dump(config, f, indent=2)
        f.truncate()
    
    print("\n✅ Done!")
    print(f"Model saved to: ./task-router/")
    print(f"Accuracy: {result['eval_accuracy']:.2%}")
