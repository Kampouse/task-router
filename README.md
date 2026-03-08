# Task Router

A lightweight DistilBERT-based classifier that routes user input to the best Ollama model.

**Accuracy:** 99.38%  
**Latency:** ~5ms  
**Model size:** 268MB

## What it does

```
User input → Task Router → Best Ollama Model
     ↓
"Debug this code" → coding → glm-5 (cloud) / qwen2.5-coder:7b (local)
"Analyze this photo" → vision → qwen3-vl:8b (cloud) / llava:7b (local)
"What is 15% of 200?" → math_reasoning → glm-5 (cloud) / deepseek-r1:7b (local)
```

## 12 Task Categories

| Category | Ollama Cloud | Ollama Local | Best For |
|----------|--------------|--------------|----------|
| coding | glm-5 | qwen2.5-coder:7b | Code, debugging |
| math_reasoning | glm-5 | deepseek-r1:7b | Math, proofs |
| vision | qwen3-vl:8b | llava:7b | Image analysis |
| research | glm-5 | qwen3:14b | Information gathering |
| tools | glm-5 | qwen3:8b | API calls, functions |
| creative | glm-5 | dolphin3:8b | Stories, creative |
| general | glm-5 | mistral:7b | Chat, small talk |
| qa | qwen3.5:9b | llama3.2:3b | Factual questions |
| summarization | qwen3.5:4b | llama3.2:3b | Summarizing text |
| translation | qwen3.5:9b | qwen2.5:7b | Language translation |
| embedding | nomic-embed-text | nomic-embed-text | Semantic search |
| cooking | qwen3.5:4b | llama3.2:3b | Recipes |

## Quick Start

```bash
# Clone and setup
git clone https://github.com/Kampouse/task-router
cd task-router
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Option 1: Download pre-trained model (recommended)
gh release download v1.0
unzip task-router.zip  # Or move model.safetensors to task-router/

# Option 2: Train from scratch
python generate_data.py    # Generate 2400 samples
python train_simple.py     # Train (~3 min)

# Test
python inference.py --benchmark
```

## Usage

### Route to Ollama Cloud

```bash
python inference.py --route "debug this async rust code"
# Task: coding
# Model: glm-5
# Confidence: 99.5%
```

### Route to Local Ollama

```bash
python inference.py --local "analyze this photo"
# llava:7b (99.9%)
```

### Just classify (no routing)

```bash
python inference.py "what is the capital of france"
# qa (99.9%)
```

### Python API

```python
from inference import classify, route

# Classify only
result = classify("debug this code")
# {'label': 'coding', 'confidence': 0.995, 'all_scores': {...}}

# Route to model
result = route("debug this code", mode="cloud")
# {'task': 'coding', 'model': 'glm-5', 'confidence': 0.995}

result = route("debug this code", mode="local")
# {'task': 'coding', 'model': 'qwen2.5-coder:7b', 'confidence': 0.995}
```

## Files

```
task-router/
├── task-router/          # Trained model (268MB)
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── config.json
├── generate_data.py      # Generate synthetic training data
├── train_simple.py       # Train the classifier
├── inference.py          # Use the classifier
├── best_params.json      # Config with model mappings
└── requirements.txt      # Dependencies
```

## Customization

### Add new task categories

1. Edit `generate_data.py`:

```python
TASK_DATA = {
    ...
    "new_category": [
        "Example prompt 1",
        "Example prompt 2",
    ],
}
```

2. Add mapping in `best_params.json`:

```json
{
  "ollama_cloud": {
    "new_category": "model-name"
  }
}
```

3. Retrain:

```bash
python generate_data.py
python train_simple.py
```

### Use different models

Edit `best_params.json`:

```json
{
  "ollama_cloud": {
    "coding": "glm-5",      // Change to any Ollama Cloud model
    "vision": "qwen3-vl:8b"
  },
  "ollama_local": {
    "coding": "qwen2.5-coder:7b"  // Change to any local model
  }
}
```

## Ollama Cloud Pricing

| Plan | Price | Best For |
|------|-------|----------|
| Free | $0 | Light usage, trying models |
| Pro | $20/mo | Day-to-day work, coding |
| Max | $100/mo | Heavy usage, agents |

See: https://ollama.com/pricing

## Model Recommendations

### GLM-5 (Ollama Cloud)
- 744B params (40B active)
- Best for: Reasoning, agentic, systems engineering
- Use for: coding, math_reasoning, research, tools

### Qwen3-VL (Ollama Cloud)
- Best vision-language model
- Use for: vision tasks

### Qwen2.5-Coder (Local)
- Best open coding model
- Use for: coding (local)

### DeepSeek-R1 (Local)
- Best open reasoning model
- Use for: math_reasoning (local)

## Performance

| Metric | Value |
|--------|-------|
| Accuracy | 99.38% |
| F1 Score | 99.37% |
| Training time | ~3 min |
| Inference latency | ~5ms |
| Model size | 268MB |

## Requirements

- Python 3.10+
- PyTorch
- Transformers
- 4GB RAM (for inference)
- 8GB RAM (for training)

## License

MIT

## Acknowledgments

- DistilBERT (HuggingFace)
- Ollama (model hosting)
- Zhipu AI (GLM-5)
- Alibaba (Qwen models)
