# Task Router

Route user input to the best Ollama model using DistilBERT classification.

**Accuracy:** 99.38%  
**Latency:** ~5ms  
**Model size:** 255MB

## What it does

```
User input → Task Router → Best Ollama Model
     ↓
"Debug this code" → coding → qwen3-coder-next (cloud) / qwen2.5-coder:7b (local)
"Analyze this photo" → vision → qwen3-vl:8b (cloud) / llava:7b (local)
"Calculate 15% of 200" → math_reasoning → glm-5 (cloud) / deepseek-r1:7b (local)
```

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

# Option 2: Train from scratch
python generate_data.py    # Generate 2400 samples
python train_simple.py     # Train (~3 min)

# Test
python inference.py --benchmark
```

## Usage

### Route with alternatives

```bash
python inference.py --alternatives "debug this code"
# Task: coding (99.5%)
#
# Cloud:
#   Primary: qwen3-coder-next
#   Alternatives: glm-4.7, minimax-m2.5, rnj-1, devstral-small-2
#
# Local:
#   qwen2.5-coder:7b
```

### Route to Ollama Cloud

```bash
python inference.py --route "debug this code"
# Task: coding
# Model: qwen3-coder-next
# Confidence: 99.5%
```

### Route to Local Ollama

```bash
python inference.py --local "analyze this photo"
# llava:7b (99.9%)
```

### Just classify

```bash
python inference.py "what is the capital of france"
# qa (99.9%)
```

## Task → Model Mapping

| Task | Cloud Primary | Cloud Alternatives | Local |
|------|---------------|-------------------|-------|
| coding | qwen3-coder-next | glm-4.7, minimax-m2.5, rnj-1, devstral-small-2 | qwen2.5-coder:7b |
| vision | qwen3-vl:8b | kimi-k2.5, ministral-3 | llava:7b |
| math_reasoning | glm-5 | deepseek-v3.2, kimi-k2-thinking | deepseek-r1:7b |
| research | qwen3.5:27b | qwen3-next, kimi-k2.5, glm-5 | qwen3:14b |
| tools | devstral-small-2 | qwen3-coder-next, nemotron-3-nano | qwen3:8b |
| creative | qwen3.5:9b | gemini-3-flash, cogito-2.1 | dolphin3:8b |
| general | qwen3.5:9b | gemini-3-flash | mistral:7b |
| qa | gemini-3-flash | qwen3.5:9b, glm-5 | llama3.2:3b |
| summarization | gemini-3-flash | qwen3.5:4b | llama3.2:3b |
| translation | minimax-m2.1 | qwen3.5:9b | qwen2.5:7b |
| cooking | qwen3.5:4b | - | llama3.2:3b |
| embedding | nomic-embed-text | - | nomic-embed-text |

## Ollama Cloud Models

| Model | Specialty | Best For |
|-------|-----------|----------|
| **qwen3-coder-next** | Coding-focused, agentic | coding |
| **qwen3-vl** | Vision-language | vision |
| **glm-5** | Reasoning, systems engineering | math_reasoning |
| **qwen3.5** | Multimodal (vision, tools, thinking) | research, creative, general |
| **devstral-small-2** | Tools, codebase exploration | tools |
| **gemini-3-flash** | Frontier intelligence, speed | qa, summarization |
| **minimax-m2.1** | Multilingual | translation |
| **deepseek-v3.2** | Reasoning, agent | math_reasoning |
| **kimi-k2.5** | Multimodal agentic | vision, research |
| **glm-4.7** | Coding | coding |

## Python API

```python
from inference import classify, route, route_with_alternatives

# Classify only
result = classify("debug this code")
# {'label': 'coding', 'confidence': 0.995}

# Route to cloud model
result = route("debug this code", mode="cloud")
# {'task': 'coding', 'model': 'qwen3-coder-next', 'confidence': 0.995}

# Route to local model
result = route("debug this code", mode="local")
# {'task': 'coding', 'model': 'qwen2.5-coder:7b', 'confidence': 0.995}

# Get alternatives
result = route_with_alternatives("debug this code")
# {
#   'task': 'coding',
#   'cloud': 'qwen3-coder-next',
#   'cloud_alternatives': ['glm-4.7', 'minimax-m2.5', ...],
#   'local': 'qwen2.5-coder:7b'
# }
```

## Ollama Cloud Pricing

| Plan | Price | Best For |
|------|-------|----------|
| Free | $0 | Light usage |
| Pro | $20/mo | Day-to-day work |
| Max | $100/mo | Heavy usage, agents |

## Performance

| Metric | Value |
|--------|-------|
| Accuracy | 99.38% |
| F1 Score | 99.37% |
| Training time | ~3 min |
| Inference | ~5ms |
| Model size | 255MB |

## License

MIT
