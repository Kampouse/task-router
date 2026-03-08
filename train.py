#!/usr/bin/env python3
"""Train DistilBERT for task classification with Optuna hyperparameter search."""

import json
import numpy as np
import optuna
from optuna.exceptions import TrialPruned
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
TASK_LABELS = ["coding", "cooking", "creative", "general", "math", "qa", "research"]
label2id = {label: i for i, label in enumerate(TASK_LABELS)}
id2label = {i: label for label, i in label2id.items()}

# Tokenizer (load once)
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")


def tokenize(batch):
    """Tokenize a batch of texts."""
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )


def encode_labels(batch):
    """Convert string labels to integers."""
    batch["label"] = label2id[batch["label"]]
    return batch


def compute_metrics(eval_pred):
    """Compute accuracy and F1 score."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
    }


def objective(trial):
    """Optuna objective function for hyperparameter optimization."""
    
    # Hyperparameter search space
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    num_epochs = trial.suggest_int("num_epochs", 2, 5)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
    
    # Load dataset
    dataset = load_dataset("json", data_files="tasks.jsonl", split="train")
    
    # Preprocess
    dataset = dataset.map(encode_labels)
    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    # Split
    splits = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = splits["train"]
    eval_dataset = splits["test"]
    
    # Model
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(TASK_LABELS),
        id2label=id2label,
        label2id=label2id,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./trial-{trial.number}",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=10,
        report_to="none",  # Disable wandb/tensorboard
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
    try:
        trainer.train()
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        raise TrialPruned()
    
    # Evaluate
    result = trainer.evaluate()
    accuracy = result["eval_accuracy"]
    
    # Report for pruning
    trial.report(accuracy, step=num_epochs)
    
    if trial.should_prune():
        raise TrialPruned()
    
    print(f"\nTrial {trial.number}: accuracy={accuracy:.4f}")
    print(f"  lr={learning_rate:.2e}, batch={batch_size}, epochs={num_epochs}")
    
    # Save best model
    if trial.number == 0 or accuracy > study.best_value:
        trainer.save_model("./task-router-best")
        tokenizer.save_pretrained("./task-router-best")
        print(f"  → Saved as best model!")
    
    return accuracy


if __name__ == "__main__":
    print("=" * 60)
    print("Task Router Training with Optuna")
    print("=" * 60)
    print(f"Labels: {TASK_LABELS}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'}")
    print()
    
    # Run Optuna study
    study = optuna.create_study(
        direction="maximize",
        study_name="task-router",
        storage="sqlite:///task-router.db",
        load_if_exists=True,
    )
    
    # Run trials
    study.optimize(
        objective,
        n_trials=20,
        timeout=3600,  # 1 hour max
        show_progress_bar=True,
    )
    
    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Best accuracy: {study.best_value:.4f}")
    print(f"Best params: {json.dumps(study.best_params, indent=2)}")
    
    # Save best params
    with open("best_params.json", "w") as f:
        json.dump({
            "accuracy": study.best_value,
            "params": study.best_params,
            "labels": TASK_LABELS,
        }, f, indent=2)
    
    print("\nSaved to:")
    print("  - ./task-router-best/  (model)")
    print("  - best_params.json     (hyperparameters)")
