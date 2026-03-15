"""Dataset preparation utilities.

Functions for loading and preprocessing the Flickr30k dataset into a
format suitable for fine-tuning T5-family models on storytelling tasks.
"""

import os

import pandas as pd
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split


def prepare_storytelling_dataset(
    dataset_name="nlphuji/flickr30k",
    split="test[:10]",
    image_dir="images_flickr30k",
    test_size=0.2,
    random_state=42,
):
    """Prepare a storytelling dataset from Flickr30k.

    Downloads a subset of images and their captions, then augments the
    data by creating entries for each story style.

    Parameters
    ----------
    dataset_name : str
        Hugging Face dataset identifier.
    split : str
        Dataset split specification (e.g. ``"test[:10]"``).
    image_dir : str
        Directory where downloaded images are saved.
    test_size : float
        Fraction of data reserved for testing.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[Dataset, Dataset]
        Training and testing :class:`datasets.Dataset` objects.
    """
    try:
        dataset = load_dataset(dataset_name, split=split)
    except Exception as exc:
        print(f"Error loading dataset: {exc}")
        empty = Dataset.from_dict(
            {"image_path": [], "reference_story": [], "style": []}
        )
        return empty, empty

    os.makedirs(image_dir, exist_ok=True)

    examples = []
    for idx, item in enumerate(dataset):
        image = item["image"]
        if image is None or not hasattr(image, "save"):
            continue

        file_path = os.path.join(image_dir, f"image_{idx}.jpg")
        try:
            image.save(file_path)
        except Exception as exc:
            print(f"Error saving image {file_path}: {exc}")
            continue

        captions = item["caption"]
        if not captions:
            continue

        examples.append({
            "image_path": file_path,
            "reference_story": captions[0],
        })

    # Augment with different story styles
    styles = ["creative", "factual", "emotional", "concise"]
    augmented = []
    for ex in examples:
        for style in styles:
            augmented.append({**ex, "style": style})

    if not augmented:
        empty = Dataset.from_dict(
            {"image_path": [], "reference_story": [], "style": []}
        )
        return empty, empty

    train_data, test_data = train_test_split(
        augmented, test_size=test_size, random_state=random_state
    )

    return (
        Dataset.from_pandas(pd.DataFrame(train_data)),
        Dataset.from_pandas(pd.DataFrame(test_data)),
    )


def build_finetune_dataset(
    story_generator,
    dataset,
    tokenizer,
    max_input_len=128,
    max_target_len=64,
):
    """Build a tokenised dataset for fine-tuning a seq2seq LLM.

    Parameters
    ----------
    story_generator : MultimodalStoryGenerator
        Instance used to create prompts from images.
    dataset : Dataset
        Input dataset with ``image_path``, ``style``, and
        ``reference_story`` columns.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer of the model to be fine-tuned.
    max_input_len : int
        Maximum token length for input prompts.
    max_target_len : int
        Maximum token length for target stories.

    Returns
    -------
    Dataset
        A dataset with ``input_ids``, ``attention_mask``, and ``labels``
        columns ready for :class:`trl.SFTTrainer`.
    """
    records = []
    for row in dataset:
        try:
            if not os.path.exists(row["image_path"]):
                print(f"Image path {row['image_path']} does not exist. Skipping.")
                continue

            analysis = story_generator.analyze_image(row["image_path"])
            prompt = story_generator._create_enhanced_prompt(
                analysis, row["style"]
            ).strip()
            story = str(row["reference_story"])

            input_enc = tokenizer(
                prompt,
                truncation=True,
                padding="max_length",
                max_length=max_input_len,
                return_tensors="pt",
            )
            target_enc = tokenizer(
                story,
                truncation=True,
                padding="max_length",
                max_length=max_target_len,
                return_tensors="pt",
            )

            records.append({
                "input_ids": input_enc["input_ids"].squeeze().tolist(),
                "attention_mask": input_enc["attention_mask"].squeeze().tolist(),
                "labels": target_enc["input_ids"].squeeze().tolist(),
            })
        except Exception as exc:
            print(
                f"Error processing {row.get('image_path', 'unknown')}: {exc}"
            )

    if not records:
        return Dataset.from_dict(
            {"input_ids": [], "attention_mask": [], "labels": []}
        )

    return Dataset.from_pandas(pd.DataFrame(records))
