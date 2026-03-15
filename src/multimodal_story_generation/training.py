"""Fine-tuning utilities using LoRA and SFTTrainer.

Provides a high-level function for fine-tuning T5-family models on the
storytelling task using parameter-efficient LoRA adapters.
"""

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


def fine_tune_llm(
    model_name,
    train_dataset,
    eval_dataset,
    story_generator_instance,
    max_input_len=128,
    max_target_len=64,
    output_dir="results_finetune",
    num_train_epochs=3,
    learning_rate=2e-4,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    lora_r=8,
    lora_alpha=32,
    lora_dropout=0.05,
):
    """Fine-tune a T5-family model with LoRA.

    Parameters
    ----------
    model_name : str
        Key in ``story_generator_instance.llms`` (e.g. ``"flan-t5-base"``).
    train_dataset : datasets.Dataset
        Tokenised training dataset produced by
        :func:`~multimodal_story_generation.data.build_finetune_dataset`.
    eval_dataset : datasets.Dataset
        Tokenised evaluation dataset.
    story_generator_instance : MultimodalStoryGenerator
        Instance used to resolve the model identifier.
    max_input_len : int
        Maximum input sequence length.
    max_target_len : int
        Maximum target sequence length.
    output_dir : str
        Directory to write checkpoints.
    num_train_epochs : int
        Number of training epochs.
    learning_rate : float
        Peak learning rate.
    per_device_train_batch_size : int
        Per-device training batch size.
    gradient_accumulation_steps : int
        Number of gradient accumulation steps.
    lora_r : int
        Rank of the LoRA matrices.
    lora_alpha : int
        LoRA scaling factor.
    lora_dropout : float
        Dropout probability for LoRA layers.

    Returns
    -------
    tuple
        ``(model, tokenizer, training_args, log_history,
        train_losses, eval_losses, epochs_log)``
    """
    if model_name not in story_generator_instance.llms:
        raise ValueError(
            f"Model '{model_name}' not found in story_generator_instance.llms."
        )

    # Resolve Hugging Face model identifier
    if "flan" in model_name or ("t5" in model_name and model_name != "t5-small"):
        base_model_id = f"google/{model_name}"
    else:
        base_model_id = model_name

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_id, device_map="auto"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q", "v"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=f"{output_dir}/{model_name}",
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        logging_steps=5,
        save_steps=50,
        fp16=True,
        do_eval=True,
        eval_steps=20,
        report_to=["tensorboard"],
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    log_history = trainer.state.log_history
    train_losses = [e["loss"] for e in log_history if "loss" in e]
    eval_losses = [e["eval_loss"] for e in log_history if "eval_loss" in e]
    epochs_log = [e["epoch"] for e in log_history if "epoch" in e]

    return (
        model,
        tokenizer,
        training_args,
        log_history,
        train_losses,
        eval_losses,
        epochs_log,
    )
