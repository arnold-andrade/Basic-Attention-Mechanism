# Attention Mechanism — PyTorch Notebook Demo

This repository contains a compact, Jupyter Notebook that demonstrates a simple attention mechanism implemented in PyTorch.

## Overview

The main file, `Attention_Mechanism.ipynb`, defines a lightweight `Attention` module and runs a small example with randomly-initialized encoder outputs and a decoder hidden state. The notebook prints the encoder outputs, decoder hidden state, attention weights, and the resulting context vector so you can inspect how attention distributes importance across time steps.

## Purpose

- Teach the core idea of attention in sequence models (how a decoder focuses on encoder timesteps).
- Show a minimal, readable PyTorch implementation that illustrates tensor shapes and operations.
- Provide a hands-on example users can run and extend into seq2seq models.

## What You'll Learn

- How to compute attention energy scores by combining decoder hidden state and encoder outputs.
- How to normalize scores with softmax to produce attention weights.
- How to compute a context vector as a weighted sum of encoder outputs.

## Files

- `Attention_Mechanism.ipynb` — Jupyter Notebook with the implementation and runnable example.

## Requirements

- Python 3.8+ (or compatible)
- PyTorch (install from https://pytorch.org for appropriate CUDA/CPU wheels)
- Jupyter Notebook

## Quick Start

1. (Optional) Create and activate a virtual environment:

```powershell
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
```

2. Install dependencies:

```powershell
pip install torch jupyter
```

3. Launch Jupyter and open the notebook:

```powershell
jupyter notebook Attention_Mechanism.ipynb
```

4. Run the cells to see printed encoder outputs, decoder hidden state, attention weights, and the computed context vector.

> Note: For GPU-enabled PyTorch builds, follow the install instructions at https://pytorch.org.

## Brief Implementation Notes

- The notebook defines `Attention(nn.Module)` with `self.attn = nn.Linear(hidden_dim * 2, hidden_dim)` and a learnable vector `self.v` to score the transformed concatenated states.
- In `forward`, it repeats the decoder hidden state across the sequence length, concatenates it with encoder outputs, applies a `tanh`-activated linear layer, and uses a batched dot product with `v` to get raw scores.
- Scores are normalized with `softmax` to form attention weights, and the context vector is the weighted sum of encoder outputs.

## Expected Outcome

Running the notebook prints the intermediate tensors and shows how attention assigns weights to encoder timesteps and produces a context vector that the decoder can use for prediction.


