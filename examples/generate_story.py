"""Example: generate a story from a local image file.

Usage
-----
    python examples/generate_story.py path/to/image.jpg

This script demonstrates the two-stage pipeline:
1. Image captioning with BLIP.
2. Story generation with a T5-family model.
"""

import sys

from multimodal_story_generation import MultimodalStoryGenerator


def main():
    if len(sys.argv) < 2:
        print("Usage: python examples/generate_story.py <image_path> [style]")
        print("Styles: creative, factual, emotional, concise")
        sys.exit(1)

    image_path = sys.argv[1]
    style = sys.argv[2] if len(sys.argv) > 2 else "creative"

    print("Loading models (this may take a moment on first run)...")
    generator = MultimodalStoryGenerator()

    print(f"\nAnalysing image: {image_path}")
    analysis = generator.analyze_image(image_path)
    print(f"Caption: {analysis['description']}")

    print(f"\nGenerating story (style={style}, model=flan-t5-base)...")
    story = generator.generate_story(analysis, llm_choice="flan-t5-base", style=style)

    print("\n--- Generated Story ---")
    print(story)


if __name__ == "__main__":
    main()
