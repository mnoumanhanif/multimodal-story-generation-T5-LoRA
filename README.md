# 🧠 Multimodal Story Generation with T5 + LoRA

[![CI](https://github.com/mnoumanhanif/multimodal-story-generation-T5-LoRA/actions/workflows/ci.yml/badge.svg)](https://github.com/mnoumanhanif/multimodal-story-generation-T5-LoRA/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

Generate **human-like narrative stories from images** using a multimodal
AI pipeline that combines image captioning (BLIP) with T5-family language
models, enhanced through **LoRA fine-tuning**.

> Developed by **Muhammad Nouman Hanif** and **Muhammad Sabeel Ahmed**
> — FAST-NUCES | Supervised by **Dr. Jawwad Ahmed Shamsi**

🔗 [Open the interactive notebook in Google Colab](https://colab.research.google.com/drive/1y8wSgmJiSngF034GE9J_-bGV_gF6r6NJ?usp=sharing)

---

## ✨ Key Features

- 🖼 **Image captioning** with [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base)
- ✍️ **Story generation** using T5 / Flan-T5 language models
- 🧠 **LoRA fine-tuning** for parameter-efficient style transfer
- 📊 **Comprehensive evaluation**: BLEU, ROUGE-L, BERTScore, perplexity, lexical diversity
- 🎨 **Multiple story styles**: creative, factual, emotional, concise

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Image captioning | BLIP (Salesforce) |
| Language models | T5-small, Flan-T5-small, Flan-T5-base |
| Fine-tuning | LoRA via PEFT + SFTTrainer (TRL) |
| Framework | PyTorch, Hugging Face Transformers |
| Evaluation | BERTScore, ROUGE, NLTK BLEU |
| Visualization | Matplotlib, Seaborn |

---

## 📁 Project Structure

```
.
├── src/multimodal_story_generation/   # Core Python package
│   ├── __init__.py
│   ├── pipeline.py                    # MultimodalStoryGenerator class
│   ├── data.py                        # Dataset preparation (Flickr30k)
│   ├── evaluation.py                  # Metrics (BLEU, ROUGE, BERTScore, …)
│   ├── training.py                    # LoRA fine-tuning
│   └── visualization.py              # Comparison plots
├── notebooks/                         # Jupyter notebook (full pipeline)
├── tests/                             # Unit tests
├── configs/                           # Default configuration
├── examples/                          # Usage examples
├── docs/                              # Documentation
│   ├── setup.md
│   ├── architecture.md
│   └── development.md
├── .github/                           # CI and templates
├── pyproject.toml                     # Build & dependency config
├── requirements.txt                   # Flat dependency list
├── CONTRIBUTING.md
├── CHANGELOG.md
├── LICENSE
└── README.md
```

---

## 🚀 Installation

### Prerequisites

- Python 3.9+
- pip
- (Recommended) CUDA-capable GPU

### Quick Start

```bash
git clone https://github.com/mnoumanhanif/multimodal-story-generation-T5-LoRA.git
cd multimodal-story-generation-T5-LoRA
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

See [docs/setup.md](docs/setup.md) for detailed setup instructions.

---

## 📖 Usage

### Python API

```python
from multimodal_story_generation import MultimodalStoryGenerator

generator = MultimodalStoryGenerator()

# Stage 1: Analyze an image
analysis = generator.analyze_image("photo.jpg")
print(analysis["description"])

# Stage 2: Generate a story
story = generator.generate_story(analysis, llm_choice="flan-t5-base", style="creative")
print(story)
```

### Command-Line Example

```bash
python examples/generate_story.py photo.jpg creative
```

### Jupyter Notebook

The full end-to-end pipeline (training, evaluation, and visualisation)
is available in `notebooks/Multimodal_Story_Generation.ipynb`.

---

## 🧪 Testing

```bash
pytest tests/ -v
```

---

## 🔍 Linting

```bash
flake8 src/ tests/ --max-line-length=100
```

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md)
for guidelines on how to get started.

---

## 📝 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE)
for details.

---

## 👥 Authors & Maintainers

- **Muhammad Nouman Hanif** — [GitHub](https://github.com/mnoumanhanif)
- **Muhammad Sabeel Ahmed**

Supervised by **Dr. Jawwad Ahmed Shamsi** — FAST-NUCES
