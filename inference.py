#!/usr/bin/env python3
"""Inference script for task classification with Ollama routing."""

import json
import torch
from transformers import pipeline

# Load config
try:
    with open("best_params.json") as f:
        config = json.load(f)
        LABELS = config.get("labels", ["coding", "cooking", "creative", "embedding", "general", "math_reasoning", "qa", "research", "summarization", "tools", "translation", "vision"])
        OLLAMA_CLOUD = config.get("ollama_cloud", {})
        OLLAMA_CLOUD_ALT = config.get("ollama_cloud_alternatives", {})
        OLLAMA_LOCAL = config.get("ollama_local", {})
        MODEL_DETAILS = config.get("model_details", {})
except FileNotFoundError:
    LABELS = ["coding", "cooking", "creative", "embedding", "general", "math_reasoning", "qa", "research", "summarization", "tools", "translation", "vision"]
    OLLAMA_CLOUD = {}
    OLLAMA_CLOUD_ALT = {}
    OLLAMA_LOCAL = {}
    MODEL_DETAILS = {}

# Load model
classifier = pipeline(
    "text-classification",
    model="./task-router",
    tokenizer="./task-router",
    top_k=None,  # Return all scores
)


def classify(text: str) -> dict:
    """Classify a text into a task category.
    
    Returns:
        dict with 'label', 'confidence', and 'all_scores' keys
    """
    results = classifier(text)[0]
    
    # Get best prediction
    best = max(results, key=lambda x: x["score"])
    
    return {
        "label": best["label"],
        "confidence": round(best["score"], 4),
        "all_scores": {r["label"]: round(r["score"], 4) for r in results}
    }


def route(text: str, mode: str = "cloud") -> dict:
    """Classify text and return the best Ollama model.
    
    Args:
        text: Input text to classify
        mode: "cloud" for Ollama Cloud, "local" for local Ollama
    
    Returns:
        dict with 'task', 'confidence', 'model', and 'all_scores'
    """
    result = classify(text)
    task = result["label"]
    
    # Get model mapping
    if mode == "cloud":
        model = OLLAMA_CLOUD.get(task, "qwen3.5:9b")
    else:
        model = OLLAMA_LOCAL.get(task, "mistral:7b")
    
    return {
        "task": task,
        "confidence": result["confidence"],
        "model": model,
        "all_scores": result["all_scores"]
    }


def route_with_alternatives(text: str) -> dict:
    """Classify text and return primary + alternative models.
    
    Returns:
        dict with 'task', 'confidence', 'cloud', 'cloud_alternatives', 'local'
    """
    result = classify(text)
    task = result["label"]
    
    return {
        "task": task,
        "confidence": result["confidence"],
        "cloud": OLLAMA_CLOUD.get(task, "qwen3.5:9b"),
        "cloud_alternatives": OLLAMA_CLOUD_ALT.get(task, []),
        "local": OLLAMA_LOCAL.get(task, "mistral:7b"),
        "all_scores": result["all_scores"]
    }


def interactive_mode():
    """Interactive CLI for testing."""
    print("Task Router - Interactive Mode")
    print("Enter text to classify (or 'quit' to exit)")
    print("-" * 50)
    
    while True:
        text = input("\n> ").strip()
        
        if text.lower() in ["quit", "exit", "q"]:
            break
        
        if not text:
            continue
        
        result = classify(text)
        
        print(f"\n  Task: {result['label']}")
        print(f"  Confidence: {result['confidence']:.1%}")
        
        # Show top 3
        sorted_scores = sorted(
            result["all_scores"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        print("  Top 3:")
        for label, score in sorted_scores:
            bar = "█" * int(score * 20)
            print(f"    {label:10} {score:.1%} {bar}")


def benchmark():
    """Run benchmark on test cases."""
    test_cases = [
        ("How do I fix this Python bug?", "coding"),
        ("Recipe for chocolate cake", "cooking"),
        ("Research AI trends", "research"),
        ("What is 15% of 200?", "math_reasoning"),
        ("Write a sci-fi story", "creative"),
        ("What is the capital of France?", "qa"),
        ("Hello, how are you?", "general"),
        ("What's in this image?", "vision"),
        ("Find similar documents", "embedding"),
        ("Call the weather API", "tools"),
        ("Translate this to Spanish", "translation"),
        ("Summarize this article", "summarization"),
    ]
    
    print("\nBenchmark Results (Ollama Cloud):")
    print("-" * 60)
    
    correct = 0
    for text, expected in test_cases:
        result = route(text, mode="cloud")
        is_correct = result["task"] == expected
        correct += is_correct
        status = "✓" if is_correct else "✗"
        
        print(f"{status} '{text[:40]}...'")
        print(f"  Task: {result['task']} → Model: {result['model']} ({result['confidence']:.1%})")
    
    print("-" * 60)
    print(f"Accuracy: {correct}/{len(test_cases)} ({correct/len(test_cases):.0%})")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--benchmark":
            benchmark()
        elif sys.argv[1] == "--route":
            # Route with model recommendation
            text = " ".join(sys.argv[2:])
            result = route(text, mode="cloud")
            print(f"Task: {result['task']}")
            print(f"Model: {result['model']}")
            print(f"Confidence: {result['confidence']:.1%}")
        elif sys.argv[1] == "--alternatives":
            # Show primary + alternatives
            text = " ".join(sys.argv[2:])
            result = route_with_alternatives(text)
            print(f"Task: {result['task']} ({result['confidence']:.1%})")
            print(f"\nCloud:")
            print(f"  Primary: {result['cloud']}")
            if result['cloud_alternatives']:
                print(f"  Alternatives: {', '.join(result['cloud_alternatives'])}")
            print(f"\nLocal:")
            print(f"  {result['local']}")
        elif sys.argv[1] == "--local":
            # Route to local model
            text = " ".join(sys.argv[2:])
            result = route(text, mode="local")
            print(f"{result['model']} ({result['confidence']:.1%})")
        else:
            # Classify from command line (task only)
            text = " ".join(sys.argv[1:])
            result = classify(text)
            print(f"{result['label']} ({result['confidence']:.1%})")
    else:
        interactive_mode()
