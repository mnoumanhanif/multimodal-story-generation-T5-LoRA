# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] — 2026-03-15

### Added

- Modular Python package under `src/multimodal_story_generation/`.
- `MultimodalStoryGenerator` class for image-to-story generation.
- Dataset preparation utilities for Flickr30k.
- Evaluation metrics: BLEU, ROUGE-L, BERTScore, perplexity, diversity.
- LoRA fine-tuning via `SFTTrainer`.
- Visualisation helpers for model comparison.
- Unit tests for evaluation and pipeline modules.
- GitHub Actions CI workflow.
- Issue and pull request templates.
- Documentation (`docs/setup.md`, `docs/architecture.md`, `docs/development.md`).
- `CONTRIBUTING.md` and `CHANGELOG.md`.
- `pyproject.toml` with pinned dependencies.
- Default configuration in `configs/default.yaml`.
- Usage example script in `examples/generate_story.py`.
