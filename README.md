# 🌍 Climate Intelligence: Heatwave Early Warning & Monitoring with Qwen 2.5 (3B)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Unsloth](https://img.shields.io/badge/Fine--Tuned%20with-Unsloth-orange)](https://github.com/unslothai/unsloth)
[![Dataset](https://img.shields.io/badge/Dataset-EarthVerse-green)](https://huggingface.co/datasets/miracle10/EarthVerse)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A parameter-efficient fine-tuned (PEFT/LoRA) large language model designed for automated climate hazard intelligence, heatwave prediction, and early warning assessment using 4-bit quantized **Qwen 2.5 (3B)** and the **EarthVerse** climate benchmark.

---

## 📌 Key Features

- **Low-VRAM Training:** Fine-tuned on a single NVIDIA T4 GPU (16 GB VRAM) using 4-bit NormalFloat (NF4) quantization via Unsloth.
- **Structured Hazard Assessment:** Processes qualitative weather context and temperature anomaly observations to extract structured hazard types, event titles, and affected geographical regions.
- **Response-Only Loss Masking:** Trained specifically on early warning outputs to eliminate prompt repetition and ensure deterministic hazard classification.
- **High Evaluation Precision:** Achieves **>96% BLEU and ROUGE** scores on held-out evaluation tasks from the EarthVerse benchmark.

---

## 📊 Benchmark & Evaluation Results

Evaluated on 55 held-out test instances (`tasks[350:405]`) from the `miracle10/EarthVerse` dataset:

| Evaluation Metric | Score | Metric Description |
| :--- | :---: | :--- |
| **ROUGE-1** | **98.24%** | Unigram lexical overlap with reference assessments |
| **ROUGE-2** | **96.88%** | Bigram sequence alignment across structured metadata |
| **ROUGE-L** | **98.13%** | Longest common subsequence preservation |
| **BLEU Score** | **96.64%** | N-gram precision match against ground truth |

---

## 🏗️ Architecture & Hyperparameters

- **Base Model:** `unsloth/Qwen2.5-3B-Instruct-bnb-4bit`
- **PEFT Method:** Low-Rank Adaptation (LoRA)
- **LoRA Hyperparameters:** $r = 32$, $\alpha = 32$, Dropout = 0
- **Target Modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Trainable Parameters:** 29.9M / 3.11B (0.96% of total model capacity)
- **Optimizer:** 8-bit AdamW (`adamw_8bit`)
- **Learning Rate Schedule:** `2e-4` with Cosine Decay & 10 warmup steps
- **Dataset:** `miracle10/EarthVerse` (Split: `tasks`)

---

## 🚀 Quickstart & Setup

### 1. Run in Google Colab
Open the provided `.ipynb` notebook directly in Google Colab with a **T4 GPU** runtime enabled.

### 2. Local Environment Installation

```bash
# Clone the repository
git clone [https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git](https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git)
cd YOUR-REPO-NAME

# Install dependencies
pip install "unsloth[colab-new] @ git+[https://github.com/unslothai/unsloth.git](https://github.com/unslothai/unsloth.git)"
pip install --no-deps "xformers" "trl" peft accelerate bitsandbytes
pip install datasets evaluate rouge_score sacrebleu
