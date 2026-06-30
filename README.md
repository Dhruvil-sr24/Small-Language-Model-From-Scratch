# Small Language Model with Multi-Head Latent Attention (MLA)

Reproduce and extend modern decoder-only LLM architecture by replacing standard multi-head attention with DeepSeek-V2 Multi-Head Latent Attention to study memory-efficient inference and training dynamics. This project focuses on integrating advanced architectural improvements, particularly Multi-Head Latent Attention (MLA), to build a highly efficient LLM. The model was trained on the TinyStories dataset to demonstrate efficient architecture design and training pipelines.

## Key Features

- **Multi-Head Latent Attention (MLA)**: Implemented DeepSeek-V2 style attention with low-rank Key-Value (KV) compression. This reduces the KV cache memory footprint during inference by ~81% compared to standard Multi-Head Attention (MHA), without sacrificing model quality.
- **Architectural Optimizations**: 
  - **Rotary Position Embeddings (RoPE)** with a base of 500,000 for robust context generalization.
  - **SwiGLU** activation functions in the MLP layers for improved learning dynamics and capacity.
  - **RMSNorm** applied throughout the network for faster and more stable normalization compared to standard LayerNorm.
  - **muP (Maximal Update Parametrization)** initialization to ensure stable learning rates and transfer of hyperparameters across different model scales.
- **Efficient Training Pipeline**: Custom PyTorch training loop optimized for CUDA, handling dataset streaming, gradient accumulation, and batching efficiently.

## Model Architecture Summary

| Parameter | Configuration |
|-----------|---------------|
| Total Parameters | 114.1M |
| Layers | 24 |
| Model Dimension (`d_model`) | 512 |
| Attention Heads | 8 |
| KV Compression Rank | 128 |
| Q Compression Rank | 256 |
| MLP Inner Dimension (`d_ff`) | 2048 |
| Context Length | 2048 tokens |
| Vocabulary Size | 32,768 |

## Tech Stack
- **Deep Learning**: PyTorch, Einops
- **Data & Tokenization**: Hugging Face Datasets, Tokenizers
- **Hardware**: CUDA-optimized training
