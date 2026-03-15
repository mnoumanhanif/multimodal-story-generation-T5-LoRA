# Setup Guide

This guide walks you through setting up the Multimodal Story Generation
project on your local machine.

## Prerequisites

- Python 3.9 or later
- pip (Python package manager)
- A CUDA-capable GPU is recommended for training and inference, but
  the pipeline also works on CPU.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/mnoumanhanif/multimodal-story-generation-T5-LoRA.git
cd multimodal-story-generation-T5-LoRA
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
# venv\Scripts\activate    # Windows
```

### 3. Install Dependencies

Install the package in editable mode with development extras:

```bash
pip install -e ".[dev]"
```

Or install runtime dependencies only:

```bash
pip install -r requirements.txt
```

### 4. Verify the Installation

```bash
python -c "from multimodal_story_generation import MultimodalStoryGenerator; print('OK')"
```

## Running on Google Colab

Open the notebook directly in Colab:

[Open in Colab](https://colab.research.google.com/drive/1y8wSgmJiSngF034GE9J_-bGV_gF6r6NJ?usp=sharing)

The notebook installs all required packages automatically via
`!pip install -q ...` cells.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `CUDA out of memory` | Reduce batch size or use CPU: `device="cpu"` |
| `ModuleNotFoundError` | Ensure you ran `pip install -e .` from the repo root |
| Dataset download fails | Check your internet connection and Hugging Face access |
