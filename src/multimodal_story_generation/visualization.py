"""Visualisation helpers for model comparison.

Functions for plotting evaluation metrics across different models and
story styles.
"""

import matplotlib.pyplot as plt
import seaborn as sns

METRICS = ["bleu", "rougeL", "bertscore_f1", "perplexity", "diversity"]


def compare_llms(evaluation_results_df):
    """Generate comparison plots and a summary table.

    Parameters
    ----------
    evaluation_results_df : pandas.DataFrame
        DataFrame containing evaluation metrics (columns include
        ``model``, ``style``, and the standard metric names).

    Returns
    -------
    pandas.DataFrame or None
        Summary table grouped by model and style, or ``None`` if
        the input is empty.
    """
    if evaluation_results_df.empty:
        print("Evaluation results DataFrame is empty.")
        return None

    sns.set(style="whitegrid")

    num_metrics = len(METRICS)
    plt.figure(figsize=(18, 12))

    for i, metric in enumerate(METRICS, 1):
        plt.subplot((num_metrics + 1) // 2, 2, i)

        has_styles = (
            "style" in evaluation_results_df.columns
            and evaluation_results_df["style"].nunique() > 1
        )

        if has_styles:
            try:
                pivot = evaluation_results_df.pivot_table(
                    index="model", columns="style", values=metric
                )
                pivot.plot(kind="bar", ax=plt.gca(), width=0.8)
                plt.title(f"{metric.upper()} by Model and Style")
                plt.legend(title="Style")
            except Exception:
                sns.barplot(
                    x="model", y=metric, data=evaluation_results_df,
                    ax=plt.gca(), capsize=0.1,
                )
                plt.title(f"Average {metric.upper()} by Model")
        else:
            sns.barplot(
                x="model", y=metric, data=evaluation_results_df,
                ax=plt.gca(), capsize=0.1,
            )
            plt.title(f"Average {metric.upper()} by Model")

        plt.xticks(rotation=45, ha="right")
        plt.ylabel(metric.capitalize())

    plt.tight_layout(pad=3.0)
    plt.savefig("llm_comparison_metrics.png")
    plt.show()

    # Summary table
    try:
        summary = evaluation_results_df.groupby(
            ["model", "style"]
        )[METRICS].mean().unstack()
    except KeyError:
        summary = evaluation_results_df.groupby("model")[METRICS].mean()

    print("\nSummary Table (Mean Scores):")
    print(summary)
    return summary


def plot_training_history(
    train_losses,
    eval_losses,
    training_args,
    title="Training and Evaluation Loss",
):
    """Plot training and evaluation loss curves.

    Parameters
    ----------
    train_losses : list[float]
        Training losses recorded at each logging step.
    eval_losses : list[float]
        Evaluation losses recorded at each eval step.
    training_args : transformers.TrainingArguments
        Training arguments (used to derive x-axis steps).
    title : str
        Plot title.
    """
    plt.figure(figsize=(10, 6))

    if train_losses:
        steps = [
            i * training_args.logging_steps for i in range(len(train_losses))
        ]
        plt.plot(steps, train_losses, label="Training Loss", marker="o")

    if eval_losses:
        steps = [
            i * training_args.eval_steps for i in range(len(eval_losses))
        ]
        plt.plot(steps, eval_losses, label="Evaluation Loss", marker="x")

    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
