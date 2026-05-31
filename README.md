# GPT-2 from Scratch

A ground-up implementation of the GPT-2 large language model architecture in PyTorch, built step-by-step across modular Jupyter notebooks — from raw tokenization to instruction fine-tuning and evaluation.

---

## Overview

This project reproduces the full GPT-2 pipeline without relying on high-level abstractions, making every design decision explicit and inspectable. Each notebook isolates one concept so the implementation can be studied, modified, or extended independently.

The project covers the complete lifecycle of a transformer-based LLM:

- **Tokenization** — custom tokenizer and Byte Pair Encoding (BPE)
- **Embeddings** — token, positional, and vector embeddings
- **Attention** — scaled dot-product, causal masking, trainable self-attention, and multi-head attention
- **Architecture** — GPT-2 transformer blocks (layer norm, feed-forward, residual connections)
- **Data pipeline** — sliding-window data loaders for pretraining and fine-tuning
- **Pretraining** — training loop with cross-entropy loss, training/validation loss tracking
- **Decoding** — greedy, temperature sampling, top-k, and nucleus (top-p) strategies
- **Weight loading** — loading OpenAI's pretrained GPT-2 weights into the custom architecture
- **Fine-tuning** — classification fine-tuning (spam detection) and instruction fine-tuning
- **Evaluation** — fine-tuned LLM evaluation pipeline

---

## Notebooks

Each notebook is self-contained and builds on the previous one.

| # | Notebook | Concept |
|---|----------|---------|
| 1 | `Tokenizer.ipynb` | Custom character/word tokenizer |
| 2 | `Byte_Pair_Encoding.ipynb` | BPE tokenization from scratch |
| 3 | `Token_Embeddings.ipynb` | Token embedding layer |
| 4 | `Position Embeddings.ipynb` | Learned positional encodings |
| 5 | `Vector_embedding.ipynb` | Embedding space intuition |
| 6 | `Attention mechanism.ipynb` | Scaled dot-product attention |
| 7 | `Causal Attention.ipynb` | Autoregressive masking |
| 8 | `Trainable Self Attention.ipynb` | Q/K/V projections, multi-head attention |
| 9 | `LLM Architecture.ipynb` | Full GPT-2 block: LayerNorm, FFN, residuals |
| 10 | `LLM_Data_Preprocessing.ipynb` | Text tokenization and batching |
| 11 | `Data_Loader_Input_Output_Pairs.ipynb` | Sliding-window input-target pairs |
| 12 | `LLM Pretraining.ipynb` | Training loop implementation |
| 13 | `LLM Loss function.ipynb` | Cross-entropy loss for next-token prediction |
| 14 | `LLM Training Validation Loss.ipynb` | Train/val loss curves and overfitting analysis |
| 15 | `LLM Decoding Strategies.ipynb` | Greedy, temperature, top-k, top-p sampling |
| 16 | `Model_Weights_Loaded.ipynb` | Loading OpenAI GPT-2 pretrained weights |
| 17 | `Classification finetuning data loading.ipynb` | Dataset prep for classification |
| 18 | `Architecture Classification finetuning.ipynb` | GPT-2 head modification for classification |
| 19 | `LLM finetuning training.ipynb` | Fine-tuning training loop |
| 20 | `Instruction fine-tuning dataset prep loading.ipynb` | Alpaca-style instruction dataset |
| 21 | `DataLoaders Instruction finetuning.ipynb` | Instruction fine-tuning data pipeline |
| 22 | `Loading pretrained weights instruction finetuning.ipynb` | Warm-start from pretrained GPT-2 |
| 23 | `Fine-tuned LLM Evaluation.ipynb` | Response quality evaluation |

---

## Architecture

The model follows the GPT-2 (small) configuration:

| Hyperparameter | Value |
|----------------|-------|
| Layers | 12 |
| Attention heads | 12 |
| Embedding dimension | 768 |
| Context length | 1024 |
| Vocabulary size | 50,257 |
| Parameters | ~124M |

Key implementation details:
- Pre-norm transformer blocks (LayerNorm before attention and FFN)
- GELU activation in feed-forward layers
- Causal (autoregressive) self-attention with masking
- Weight tying between token embedding and output projection
- Compatible with OpenAI's published GPT-2 checkpoint weights

---

## Getting Started

```bash
# Clone the repo
git clone https://github.com/Raghav0079/GPT2-from-Scratch.git
cd GPT2-from-Scratch

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install torch numpy jupyter tiktoken

# Launch Jupyter
jupyter notebook
```

Open the notebooks in order, starting from `Tokenizer.ipynb`.

### Requirements

- Python 3.9+
- PyTorch 2.x
- NumPy
- tiktoken (for BPE)
- Jupyter

---

## Fine-tuning Results

**Spam Classification** — Fine-tuned GPT-2 on the SMS Spam Collection dataset. The model adapts the final transformer block's output to a binary classification head, achieving strong accuracy on held-out validation data.

**Instruction Following** — Fine-tuned on an Alpaca-style instruction dataset using a pretrained GPT-2 warm start. Evaluated using a separate LLM judge pipeline (`Fine-tuned LLM Evaluation.ipynb`).

---

## Skills Demonstrated

- Transformer architecture implementation from first principles
- PyTorch model building, training loops, and optimization
- BPE tokenization and custom data pipeline engineering
- Decoding strategy design (temperature, top-k, top-p)
- Transfer learning: loading and adapting pretrained weights
- Fine-tuning for classification and instruction following

---

## References

- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2 paper), OpenAI, 2019
- Raschka, *Build a Large Language Model (From Scratch)*, Manning, 2024
- OpenAI GPT-2 weights: [openai/gpt-2](https://github.com/openai/gpt-2)

---

## Author

**Raghav Mishra**  
[GitHub](https://github.com/Raghav0079)
