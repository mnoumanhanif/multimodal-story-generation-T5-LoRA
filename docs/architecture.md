# Architecture

This document explains the high-level architecture of the Multimodal
Story Generation pipeline.

## Overview

The system takes an **image** as input and produces a **narrative story**
as output through a two-stage pipeline:

```
Image → BLIP (caption) → T5 / Flan-T5 (story) → Output
```

### Stage 1 — Image Captioning (BLIP)

The [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base)
vision-language model generates a textual description of the image.

### Stage 2 — Story Generation (T5 Variants)

The caption is wrapped in a structured prompt with style instructions
and passed to a T5-family language model (e.g., Flan-T5-base) which
produces the final story.

### Fine-Tuning with LoRA

[LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685) is used
for parameter-efficient fine-tuning.  Only small adapter matrices are
trained while the base model weights remain frozen.

## Module Map

```
src/multimodal_story_generation/
├── __init__.py        # Package exports
├── pipeline.py        # MultimodalStoryGenerator class
├── data.py            # Dataset loading and preparation
├── evaluation.py      # Evaluation metrics (BLEU, ROUGE, BERTScore, …)
├── training.py        # LoRA fine-tuning via SFTTrainer
└── visualization.py   # Comparison plots and summaries
```

### pipeline.py

Contains the `MultimodalStoryGenerator` class:

- `analyze_image(image_path)` — runs BLIP and returns a description dict.
- `generate_story(analysis, llm_choice, style)` — generates a story.
- `_create_enhanced_prompt(analysis, style)` — builds the prompt.

### data.py

- `prepare_storytelling_dataset()` — downloads Flickr30k images, splits
  into train / test, and augments with style variants.
- `build_finetune_dataset()` — tokenises prompts and reference stories
  for `SFTTrainer`.

### evaluation.py

- `calculate_perplexity(model, tokenizer, text)` — model perplexity.
- `calculate_diversity(text)` — unique-unigram ratio.
- `evaluate_generated_stories(results, generator)` — computes all
  metrics for a batch of results.

### training.py

- `fine_tune_llm(model_name, …)` — end-to-end LoRA fine-tuning with
  configurable hyper-parameters.

### visualization.py

- `compare_llms(df)` — bar plots grouped by model / style.
- `plot_training_history(losses, …)` — loss curves over training steps.

## Data Flow

```
┌────────────┐     caption      ┌──────────────┐     story
│   Image    │ ──────────────►  │   T5 / LLM   │ ──────────►  Output
│   (BLIP)   │                  │  (+ LoRA)     │
└────────────┘                  └──────────────┘
                                       ▲
                                       │  prompt template
                                       │  (style-dependent)
```
