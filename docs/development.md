# Development Guide

This guide is for developers who want to contribute to or extend the
project.

## Prerequisites

- Python 3.9+
- Git

## Setup

```bash
git clone https://github.com/mnoumanhanif/multimodal-story-generation-T5-LoRA.git
cd multimodal-story-generation-T5-LoRA
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Project Layout

```
.
├── src/multimodal_story_generation/   # Main package
├── tests/                             # Unit tests
├── notebooks/                         # Jupyter notebook
├── configs/                           # Default config files
├── examples/                          # Usage examples
├── docs/                              # Documentation
├── .github/                           # CI and templates
├── pyproject.toml                     # Build and dependency config
├── requirements.txt                   # Flat dependency list
├── pytest.ini                         # Pytest settings
└── README.md
```

## Running Tests

```bash
pytest tests/ -v
```

## Linting

```bash
flake8 src/ tests/ --max-line-length=100
```

## Adding a New Model

1. Add the Hugging Face model ID to `pipeline.DEFAULT_LLMS`.
2. If the model requires special loading (quantisation, etc.), update
   `training.fine_tune_llm()`.
3. Add a test in `tests/test_pipeline.py`.

## Adding a New Evaluation Metric

1. Create the metric function in `evaluation.py`.
2. Integrate it into `evaluate_generated_stories()`.
3. Add the metric name to `visualization.METRICS`.
4. Write a unit test in `tests/test_evaluation.py`.

## Configuration

Default hyper-parameters and model settings live in
`configs/default.yaml`.  You can override values at runtime by passing
keyword arguments to the relevant functions.

## Continuous Integration

The GitHub Actions workflow in `.github/workflows/ci.yml` runs on every
push and pull request to the `main` branch.  It:

1. Lints the codebase with `flake8`.
2. Runs the test suite with `pytest`.

Ensure that your changes pass both checks before opening a pull request.
