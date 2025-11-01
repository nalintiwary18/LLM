# LLM

A minimal README for the `LLM` repository (owner: nalintiwary18). This repository is implemented in Python (100% Python according to repository language composition). This README explains what the repository is, how to set it up, how to run code, and where to find / add more information.

> Note: If you have a specific file in the repository you'd like documented, tell me which file (path) and I'll add a targeted description and examples for it.

## Overview

This repository (LLM) contains Python code related to working with a small GPT-style language model implementation and experiments on text data (uses the Hugging Face `datasets` C4 dataset). The README below documents the example training script `gpt.py`, the hyperparameter combinations you tried, and which configuration is currently used as the best.

## Key features

- Pure Python codebase
- Minimal, educational GPT-style transformer implementation
- Example training loop + text generation + model save
- Uses the Hugging Face `datasets` library to stream C4 data

## Requirements

- Python 3.8+
- PyTorch (tested with recent stable versions)
- Hugging Face `datasets` (for loading C4)
- (Optional, recommended) CUDA-enabled GPU + appropriate CUDA toolkit for PyTorch
- Install with:
```bash
pip install torch datasets
```

If you have a `requirements.txt` or `pyproject.toml`, use that instead.

## Installation

Clone the repository:
```bash
git clone https://github.com/nalintiwary18/LLM.git
cd LLM
```

Create and activate a virtual environment:
```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

Install dependencies:
```bash
pip install -r requirements.txt
# or
pip install torch datasets
```

## gpt.py — file-specific documentation

Path: `gpt.py`  
Purpose: A compact, from-scratch GPT-style model implemented with PyTorch to:
- build a small transformer-like architecture,
- stream and tokenize text from the C4 dataset,
- train the model with a simple training loop,
- save model weights and generate text.

What it contains (high-level)
- Data loading and character-level encoding using a small vocabulary derived from the dataset.
- Model building blocks:
  - Head — single attention head (causal)
  - MultiHeadAttention — concatenates heads (note: projection exists but currently not applied in forward)
  - FeedForward — 2-layer MLP with dropout
  - Block — attention + feed-forward with LayerNorms used after residual sums
  - GPTLanguageModel — embeddings (token + positional), stacked Blocks, LM head, forward & generate
- Training loop with evaluation via `estimate_loss`
- Model saving to `model_weights.pth` and printing absolute path
- A small text generation demo at the end

Current configuration (the one in the `gpt.py` file) — considered the best among the combinations you tried:
- block_size (context): 64
- batch_size (physical): 128
- max_iters: 5000
- learning_rate: 3e-4
- eval_iters: 250
- n_embd (hidden dim): 384
- n_head: 8
- n_layer (depth): 8
- dropout: 0.2
- Dataset: `allenai/c4` (English, streaming)

How to run
```bash
python gpt.py
```
Notes:
- The script streams data from HF datasets. Ensure network access.
- It automatically saves `model_weights.pth` to the current working directory when finished.
- For faster training and larger batch sizes, run on a CUDA GPU. The script uses device='cuda' if available.

Inputs / Outputs
- Inputs: none required on CLI — the script downloads/streams the dataset and trains.
- Outputs: `model_weights.pth` saved in working directory, printed parameter count and a generated text sample at the end.

Important implementation notes & suggested fixes (recommended changes)
- Duplicate code: `get_batch` and train/val split are defined twice. Consolidate into one definition to avoid confusion.
- MultiHeadAttention: `self.proj` is created but never used in forward(). After concatenating heads, apply the projection and dropout, e.g.
  ```
  out = torch.cat([h(x) for h in self.heads], dim=-1)
  out = self.proj(out)
  out = self.dropout(out)
  return out
  ```
- Attention scaling: current attention scaling uses `k.shape[-1] ** - 0.5`, conventional is `q @ k.transpose(...) / sqrt(head_size)` — functionally equivalent but clearer to write `q @ k.transpose(-2, -1) / math.sqrt(head_size)`.
- LayerNorm & residuals: the implementation places LayerNorm after adding the residual (Post-LN). That is a valid design but consider using Pre-LN variants (LayerNorm before each sub-layer) for more stable training in deeper models.
- `Head.__init__` calls `super() .__init__()` — the space is harmless but inconsistent with typical style; prefer `super().__init__()`.
- `position_embedding_table` usage: `pos_emb = self.position_embedding_table(torch.arange(T, device=device))` returns shape (T, n_embd) which broadcasts to (B, T, n_embd) when added to token embeddings — this is fine, but being explicit with `.unsqueeze(0)` can improve readability.
- Dataset: streaming dataset is iterated multiple times to build vocabulary and text; streaming iterators may get exhausted or behave differently than non-streaming datasets. For reproducibility, consider loading a finite split or saving a prepared subset locally.
- Loss computation: current code flattens logits/targets; that's OK. Consider label smoothing or cross-entropy reduction options for different behavior.
- Numerical stability: consider gradient clipping and mixed precision (amp) for large batch sizes or deeper models.
- Checkpoints & resume: currently the script always trains from scratch and saves one final state. Add periodic checkpointing and an option to resume training.

Suggested small improvements to try
- Use torch.cuda.amp (automatic mixed precision) for speed and memory.
- Add gradient accumulation, argument parsing for hyperparameters, and logging (e.g., TensorBoard or Weights & Biases).
- Improve tokenization: character-level models are simple but limited; consider byte-pair (BPE) or SentencePiece if you want better sample quality with smaller models.
- Make the model config-driven (YAML or JSON) so experiments are reproducible and tracked.

Classes & functions (quick reference)
- get_batch(split) — returns (x, y) batch tensors for train/val
- Head — implements a causal attention head
- MultiHeadAttention — stacks heads; currently concatenates head outputs but should project back to embedding dim
- FeedForward — MLP block used in transformer
- Block — one transformer block: attention -> add+norm -> FF -> add+norm
- GPTLanguageModel
  - forward(index, targets=None) -> logits, (optional) loss
  - generate(index, max_new_tokens) -> extended index tensor
- estimate_loss() — evaluation across `eval_iters` batches (returns train/val mean loss)
- count_parameters(model) — helper printing parameter count

Experiments (combinations you tried)
Below are the combination variants you previously reported. The current config in `gpt.py` (listed above) is the best among these experiments.

Variant A
- n_embd (hidden dim): 768
- n_head: 12
- head size: 64
- n_layer (depth): 12
- Model size (approx): 45M parameters
- block_size (context): 128
- batch_size (physical): 64
- Effective Batch Size (EBS): 64
- learning_rate: 3e-5
- dropout: 0.2

Variant B
- n_embd: 768
- n_head: 12
- head size: 64
- n_layer: 18
- Model size: ~70M parameters
- block_size: 512
- batch_size : 16 
- EBS: 128 (via gradient accumulation)
- learning_rate: 3e-4
- dropout: 0.1

Variant C
- n_embd: 768
- n_head: 12
- head size: 64
- n_layer: 24
- Model size: ~95M parameters
- block_size: 1024
- batch_size: 32
- EBS: 256 (via gradient accumulation)
- learning_rate: 6e-4
- dropout: 0.1

These were the combinations tried; the current running configuration (the one included in gpt.py) uses the smaller model parameters listed in the "Current configuration" section and produced the best training results for you.

## Testing

If a `tests/` directory exists and `pytest` is used:
```bash
pip install pytest
pytest
```
Adjust per your CI/test framework.

## Contributing

Contributions are welcome. Suggested workflow:
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feat/your-feature
   ```
3. Make changes and commit with clear messages.
4. Open a pull request describing the changes and why they are needed.

Add coding standards and a CONTRIBUTING.md as the project grows.

## Issues & Reporting bugs

Open GitHub issues for bugs, feature requests, or questions. Provide:
- Steps to reproduce
- Environment (OS, Python version, CUDA & PyTorch versions)
- Minimal code snippet if applicable
- Expected vs. actual behavior
- Relevant hyperparameters and training logs

## License

Add a LICENSE file to the repo to specify licensing. If you want a suggestion, common choices are MIT, Apache-2.0, or GPL-3.0. Example:
```text
MIT License
```

## Next steps I performed and next I recommend
I inspected `gpt.py`, extracted the model hyperparameters and behavior, and added a focused documentation section above describing how to run it, what it does, and a set of recommended fixes and experiments. Next, I recommend:
- Confirm which variant (A/B/C) you want recorded as "previous best" and whether to mark the `gpt.py` config explicitly as "current best" (I already marked it as the current best per your note).
- If you want, I can:
  - open a PR that adds this README to the repository, or
  - create a cleaned/refactored `gpt.py` with the fixes applied (single `get_batch`, correct MultiHead projection, optional AMP and command-line args) and submit as a PR.

Tell me which of those you'd like me to do next.
