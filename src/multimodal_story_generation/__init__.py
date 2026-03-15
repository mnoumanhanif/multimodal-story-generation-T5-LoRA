"""Multimodal Story Generation with T5 + LoRA.

A pipeline for generating narrative stories from visual inputs using
image captioning (BLIP) and language model generation (T5 variants)
with LoRA fine-tuning.
"""

# Lightweight imports (no heavy ML dependencies)
from multimodal_story_generation.prompts import (  # noqa: F401
    STYLE_PROMPTS,
    calculate_diversity,
    create_enhanced_prompt,
)


def __getattr__(name):
    """Lazy-load heavy modules to keep import time low."""
    if name == "MultimodalStoryGenerator":
        from multimodal_story_generation.pipeline import MultimodalStoryGenerator
        return MultimodalStoryGenerator
    if name == "prepare_storytelling_dataset":
        from multimodal_story_generation.data import prepare_storytelling_dataset
        return prepare_storytelling_dataset
    if name == "build_finetune_dataset":
        from multimodal_story_generation.data import build_finetune_dataset
        return build_finetune_dataset
    if name == "calculate_perplexity":
        from multimodal_story_generation.evaluation import calculate_perplexity
        return calculate_perplexity
    if name == "evaluate_generated_stories":
        from multimodal_story_generation.evaluation import evaluate_generated_stories
        return evaluate_generated_stories
    if name == "fine_tune_llm":
        from multimodal_story_generation.training import fine_tune_llm
        return fine_tune_llm
    if name == "compare_llms":
        from multimodal_story_generation.visualization import compare_llms
        return compare_llms
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "STYLE_PROMPTS",
    "calculate_diversity",
    "create_enhanced_prompt",
    "MultimodalStoryGenerator",
    "prepare_storytelling_dataset",
    "build_finetune_dataset",
    "calculate_perplexity",
    "evaluate_generated_stories",
    "fine_tune_llm",
    "compare_llms",
]
