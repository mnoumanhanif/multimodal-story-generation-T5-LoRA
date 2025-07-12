# 🧠 Multimodal Story Generation with T5 + LoRA

This project explores **generating human-like stories from visual inputs** using a multimodal AI pipeline that combines image understanding and natural language generation via **T5 transformers** and **LoRA fine-tuning**.

> Developed by Muhammad Nouman Hanif and Muhammad Sabeel Ahmed  
> FAST-NUCES | Supervised by Dr. Jawwad Ahmed Shamsi

---

## 🚀 Overview

We built an end-to-end pipeline that:
- 🖼 Extracts image descriptions via a vision-language model (e.g., BLIP)
- ✍️ Feeds captions into T5-family LLMs to generate narrative stories
- 🧠 Fine-tunes LLMs with **LoRA (Low-Rank Adaptation)** for storytelling tone/style
- 📊 Evaluates performance with both quantitative and qualitative metrics

---

## 📊 Evaluation Metrics

- **BLEU**, **ROUGE**, **BERTScore**, **Perplexity**, **Lexical Diversity**
- Human evaluation on: **Creativity**, **Style Transfer**, **Factual Coherence**

---

## 🛠️ Technologies Used

- `transformers`, `datasets`, `trl`, `peft`
- `PyTorch`, `Hugging Face`, `Google Colab`, `BERTScore`, `rouge_score`

---

## 📁 File Structure

| Folder         | Description                                      |
|----------------|--------------------------------------------------|
| `notebooks/`   | Jupyter/Colab notebooks for training + eval      |
| `images/`      | Visuals: attention maps, pipeline design, samples|
| `models/`      | Saved LoRA fine-tuned model (if uploaded)        |
| `data/`        | Sample prompts/captions                          |
| `results/`     | Evaluation CSVs                                  |

---

## 🔧 Setup

```bash
git clone https://github.com/your-username/multimodal-story-generation-T5-LoRA.git
cd multimodal-story-generation-T5-LoRA
pip install -r requirements.txt
