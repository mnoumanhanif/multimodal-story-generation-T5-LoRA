"""Evaluation metrics for generated stories.

Provides functions for computing BLEU, ROUGE-L, BERTScore, perplexity,
and lexical diversity on generated text.
"""

import torch
from bert_score import score as bert_score_fn
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

from multimodal_story_generation.prompts import calculate_diversity


def calculate_perplexity(model, tokenizer, text):
    """Calculate perplexity of *text* under *model*.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        Language model used for scoring.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer associated with the model.
    text : str
        Input text.

    Returns
    -------
    float
        Perplexity score.  Returns ``inf`` for empty inputs.
    """
    if not text:
        return float("inf")

    max_len = getattr(model.config, "max_length", 512)
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_len
    ).to(model.device)

    if inputs.input_ids.size(1) == 0:
        return float("inf")

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
    return torch.exp(outputs.loss).item()


def evaluate_generated_stories(results, story_generator):
    """Evaluate generated stories against reference stories.

    Computes BLEU, ROUGE-L, BERTScore F1, perplexity, and diversity
    for each result entry.

    Parameters
    ----------
    results : list[dict]
        Each dict must contain ``model``, ``style``,
        ``reference_story``, and ``generated_story``.
    story_generator : MultimodalStoryGenerator
        Instance used to access loaded LLMs for perplexity calculation.

    Returns
    -------
    list[dict]
        Evaluation metrics for each story.
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    evaluations = []

    for item in results:
        reference = str(item["reference_story"])
        generated = str(item["generated_story"])
        model_key = item["model"]

        # Perplexity
        if model_key not in story_generator.llms:
            perplexity = float("nan")
        else:
            llm = story_generator.llms[model_key]
            perplexity = calculate_perplexity(
                llm["model"], llm["tokenizer"], reference
            )

        # BERTScore
        _, _, bert_f1 = bert_score_fn(
            [generated], [reference], lang="en", verbose=False
        )

        # ROUGE-L
        rouge_scores = scorer.score(reference, generated)
        rouge_l = rouge_scores["rougeL"].fmeasure

        # BLEU (4-gram)
        bleu = sentence_bleu(
            [reference.split()],
            generated.split(),
            weights=(0.25, 0.25, 0.25, 0.25),
        )

        # Diversity
        diversity = calculate_diversity(generated)

        evaluations.append({
            "model": model_key,
            "style": item["style"],
            "bleu": bleu,
            "rougeL": rouge_l,
            "bertscore_f1": bert_f1.mean().item(),
            "perplexity": perplexity,
            "diversity": diversity,
        })

    return evaluations
