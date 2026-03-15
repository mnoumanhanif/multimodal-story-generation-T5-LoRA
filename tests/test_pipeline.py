"""Tests for the pipeline module."""

from multimodal_story_generation.prompts import (
    create_enhanced_prompt,
    STYLE_PROMPTS,
)


class TestCreateEnhancedPrompt:
    """Unit tests for create_enhanced_prompt."""

    def test_creative_style(self):
        analysis = {"description": "A dog sitting on grass"}
        prompt = create_enhanced_prompt(analysis, "creative")
        assert "creative and imaginative" in prompt
        assert "A dog sitting on grass" in prompt

    def test_factual_style(self):
        analysis = {"description": "A sunset over the ocean"}
        prompt = create_enhanced_prompt(analysis, "factual")
        assert "factual" in prompt.lower()
        assert "A sunset over the ocean" in prompt

    def test_unknown_style_uses_default(self):
        analysis = {"description": "A cat"}
        prompt = create_enhanced_prompt(analysis, "unknown_style")
        assert "Write a story" in prompt

    def test_all_known_styles_have_prompts(self):
        for style in STYLE_PROMPTS:
            analysis = {"description": "test"}
            prompt = create_enhanced_prompt(analysis, style)
            assert "test" in prompt


class TestStylePrompts:
    """Verify STYLE_PROMPTS data."""

    def test_expected_styles_exist(self):
        for style in ("creative", "factual", "emotional", "concise"):
            assert style in STYLE_PROMPTS
