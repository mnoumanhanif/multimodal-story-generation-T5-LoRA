"""Tests for the evaluation module."""

from multimodal_story_generation.prompts import calculate_diversity


class TestCalculateDiversity:
    """Unit tests for calculate_diversity."""

    def test_all_unique(self):
        assert calculate_diversity("a b c d") == 1.0

    def test_all_same(self):
        assert calculate_diversity("a a a a") == 0.25

    def test_empty_string(self):
        assert calculate_diversity("") == 0.0

    def test_single_word(self):
        assert calculate_diversity("hello") == 1.0

    def test_mixed(self):
        # 3 unique out of 4
        assert calculate_diversity("the cat the dog") == 0.75
