"""Prompt templates and style definitions.

This module contains no heavy ML dependencies and can be imported
cheaply for testing and configuration.
"""

# Style prompts used for story generation
STYLE_PROMPTS = {
    "creative": "Write a highly creative and imaginative story",
    "factual": "Write a factual description with narrative elements",
    "emotional": "Write an emotionally engaging story",
    "concise": "Write a short and concise story",
}


def create_enhanced_prompt(image_analysis, style):
    """Build a structured prompt for the LLM.

    Parameters
    ----------
    image_analysis : dict
        Dictionary containing the ``"description"`` key.
    style : str
        The desired story style.

    Returns
    -------
    str
        Formatted prompt string.
    """
    style_instruction = STYLE_PROMPTS.get(style, "Write a story")

    return (
        f"{style_instruction} based on the following image description:\n\n"
        f"Image Description: {image_analysis['description']}\n\n"
        "Guidelines:\n"
        "1. Develop characters based on detected objects or themes "
        "in the description.\n"
        "2. Maintain a consistent tone throughout the narrative.\n"
        "3. Include clear plot development with a beginning, middle, "
        "and end.\n"
        "4. Target reading level: adult.\n"
        "5. Approximate length: 300-500 words.\n\n"
        "Story:\n"
    )


def calculate_diversity(text):
    """Calculate lexical diversity as the ratio of unique unigrams.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    float
        Diversity score between 0 and 1.
    """
    tokens = str(text).split()
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)
