"""Core multimodal story generation pipeline.

This module contains the ``MultimodalStoryGenerator`` class which combines
BLIP-based image captioning with T5-family language models for story
generation.
"""

import torch
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

from multimodal_story_generation.prompts import STYLE_PROMPTS  # noqa: F401
from multimodal_story_generation.prompts import create_enhanced_prompt

# Default LLM configurations
DEFAULT_LLMS = {
    "flan-t5-base": "google/flan-t5-base",
    "flan-t5-small": "google/flan-t5-small",
    "t5-small": "t5-small",
}


class MultimodalStoryGenerator:
    """Generate stories from images using a multimodal pipeline.

    The pipeline first uses BLIP for image captioning and then feeds
    the caption into a T5-family language model to produce a narrative
    story.

    Parameters
    ----------
    device : str, optional
        Device to run models on (``"cuda"`` or ``"cpu"``).  Defaults to
        ``"cuda"`` when available.
    llm_configs : dict, optional
        Mapping of short model names to Hugging Face model identifiers.
        Defaults to :data:`DEFAULT_LLMS`.
    """

    def __init__(self, device=None, llm_configs=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Initialize BLIP model for image captioning
        self.blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(device)

        # Initialize LLMs for story generation
        if llm_configs is None:
            llm_configs = DEFAULT_LLMS

        self.llms = {}
        for name, model_id in llm_configs.items():
            self.llms[name] = {
                "tokenizer": AutoTokenizer.from_pretrained(model_id),
                "model": AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device),
            }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_image(self, image_path):
        """Analyze an image and return a textual description via BLIP.

        Parameters
        ----------
        image_path : str
            Path to the input image file.

        Returns
        -------
        dict
            A dictionary with the key ``"description"`` containing the
            generated caption.
        """
        image = Image.open(image_path).convert("RGB")

        blip_inputs = self.blip_processor(images=image, return_tensors="pt")
        blip_inputs = {
            key: val.to(self.blip_model.device) for key, val in blip_inputs.items()
        }

        blip_output = self.blip_model.generate(**blip_inputs, max_new_tokens=100)
        description = self.blip_processor.batch_decode(
            blip_output, skip_special_tokens=True
        )[0]

        return {"description": description}

    def generate_story(self, image_analysis, llm_choice="flan-t5-base", style="creative"):
        """Generate a story from an image analysis result.

        Parameters
        ----------
        image_analysis : dict
            Dictionary with the key ``"description"`` (e.g. from
            :meth:`analyze_image`).
        llm_choice : str
            Key or Hugging Face identifier of the LLM to use.
        style : str
            Desired story style (``"creative"``, ``"factual"``,
            ``"emotional"``, or ``"concise"``).

        Returns
        -------
        str
            The generated story text.

        Raises
        ------
        ValueError
            If the chosen LLM is not loaded.
        """
        llm_name_map = {v: k for k, v in DEFAULT_LLMS.items()}
        model_key = llm_name_map.get(llm_choice, llm_choice)

        if model_key not in self.llms:
            raise ValueError(
                f"Model '{llm_choice}' (resolved to '{model_key}') not found. "
                f"Available models: {list(self.llms.keys())}"
            )

        llm = self.llms[model_key]
        tokenizer = llm["tokenizer"]
        model = llm["model"]

        prompt = self._create_enhanced_prompt(image_analysis, style)
        inputs = tokenizer(
            prompt, return_tensors="pt", max_length=512, truncation=True
        ).to(self.device)

        outputs = model.generate(
            **inputs, max_new_tokens=500, num_beams=4, early_stopping=True
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _create_enhanced_prompt(image_analysis, style):
        """Build a structured prompt for the LLM.

        Delegates to :func:`multimodal_story_generation.prompts.create_enhanced_prompt`.
        """
        return create_enhanced_prompt(image_analysis, style)
