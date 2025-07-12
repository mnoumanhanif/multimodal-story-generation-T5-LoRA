# 🧠 Multimodal Story Generation with T5 + LoRA

This project explores **generating human-like stories from visual inputs** using a multimodal AI pipeline that combines image understanding and natural language generation via **T5 transformers** and **LoRA fine-tuning**.

> Developed by Muhammad Nouman Hanif and Muhammad Sabeel Ahmed  
> FAST-NUCES | Supervised by Dr. Jawwad Ahmed Shamsi

🔗 **Run it in Colab**: (🔗 **Link**: [Open in Colab](https://colab.research.google.com/drive/1y8wSgmJiSngF034GE9J_-bGV_gF6r6NJ?usp=sharing)


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

## 🔧 Setup

```bash
git clone https://github.com/your-username/multimodal-story-generation-T5-LoRA.git
cd multimodal-story-generation-T5-LoRA
pip install -r requirements.txt
