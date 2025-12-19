# Liora: Lorentz Interpretable ODE-Regularized Attention-based VAE

**A deep learning framework for single-cell RNA-seq analysis combining geometric manifold learning, latent dynamics, and comprehensive interpretability.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/liora)](https://pypi.org/project/liora/)

Liora is a PyTorch-based variational autoencoder (VAE) for single-cell RNA-seq (AnnData) that learns low-dimensional embeddings from **raw count matrices** using count-likelihood objectives (NB/ZINB/Poisson/ZIP). It integrates (1) Lorentz/hyperbolic manifold regularization for hierarchical structure, (2) latent Neural ODE dynamics for continuous trajectory inference, and (3) an extensive interpretability suite for gene-to-latent attribution.

## Key Features

- **Advanced VAE Architecture**: Dimensionality reduction with count-based likelihoods (NB, ZINB, Poisson, ZIP)
- **Geometric Manifold Learning**: Lorentz (hyperbolic) or Euclidean regularization for hierarchical structure
- **Neural ODE Trajectories**: Latent dynamics solved with `torchdiffeq` (CPU by design). Supports ODE function types `legacy`, `time_mlp` (time-conditioned), and `gru`.
- **Comprehensive Interpretability**: Attribution analysis for Genes → Latents (discriminative) and Latents → Genes (reconstructive) pathways.
- **Flexible Encoders**: Standard MLP or Transformer-based with self-attention for capturing complex feature dependencies.
- **Information Bottleneck**: Hierarchical representation with controllable compression via the `i_dim` parameter.

## Data Requirements

- `adata.layers[layer]` must contain **raw, non-negative integer-like counts** (UMI counts).  
  Liora checks this heuristically and raises a `ValueError` if the layer looks normalized/log-transformed.
- Liora applies its own `log1p` + clipping / adaptive normalization internally for training.

## Installation

```bash
pip install liora
```

Or install from source:

```bash
git clone https://github.com/PeterPonyu/Liora.git
cd Liora
pip install -e .
```

## Quick Start

### Basic Usage

```python
import scanpy as sc
from liora import Liora

# Load your data
adata = sc.read_h5ad('data.h5ad')

# Train with default settings
model = Liora(
    adata,
    layer='counts',
    hidden_dim=128,
    latent_dim=10,
    i_dim=2,
)
model.fit(epochs=100)

# Extract embeddings
latent = model.get_latent()
```

### Advanced Configuration

```python
# Transformer encoder + Neural ODE trajectory inference
model = Liora(
    adata,
    layer='counts',
    hidden_dim=128,
    latent_dim=10,
    i_dim=2,
    # Encoder configuration
    encoder_type='transformer',
    attn_embed_dim=64,
    attn_num_heads=4,
    attn_num_layers=2,
    attn_seq_len=32,
    # ODE configuration
    use_ode=True,
    ode_type='time_mlp',
    ode_time_cond='concat',
    ode_hidden_dim=64,
    ode_solver_method='dopri5',
    ode_rtol=1e-5,
    ode_atol=1e-7,
    # Loss weights
    lorentz=5.0,
    beta=1.0,
)
model.fit(epochs=200, patience=20)

# Extract results
latent = model.get_latent()           # Latent embeddings
bottleneck = model.get_bottleneck()   # Information bottleneck
pseudotime = model.get_time()         # Predicted pseudotime
transitions = model.get_transition()  # Transition matrix
```

> Note: `get_time()` and `get_transition()` require `use_ode=True`.

## Configuration Guide

### Architecture Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_dim` | int | 128 | Hidden layer dimension |
| `latent_dim` | int | 10 | Primary latent space size |
| `i_dim` | int | 2 | Information bottleneck dimension |

### Encoder Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `encoder_type` | str | `'mlp'` | `'mlp'` or `'transformer'` |
| `attn_embed_dim` | int | 64 | Transformer embedding dimension |
| `attn_num_heads` | int | 4 | Number of attention heads |
| `attn_num_layers` | int | 2 | Transformer encoder layers |
| `attn_seq_len` | int | 32 | Token sequence length |

### ODE Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_ode` | bool | False | Enable Neural ODE |
| `ode_type` | str | `'time_mlp'` | `'legacy'`, `'time_mlp'`, or `'gru'` |
| `ode_time_cond` | str | `'concat'` | `'concat'`, `'film'`, or `'add'` |
| `ode_hidden_dim` | int | None | ODE network hidden size |
| `ode_solver_method` | str | `'rk4'` | Solver: `'rk4'`, `'dopri5'`, `'adams'`, etc. |
| `ode_step_size` | float | None | Fixed-step size or `'auto'` |
| `ode_rtol` | float | None | Relative tolerance (adaptive solvers) |
| `ode_atol` | float | None | Absolute tolerance (adaptive solvers) |

### Loss Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `recon` | float | 1.0 | Reconstruction loss weight |
| `irecon` | float | 0.0 | Bottleneck reconstruction weight |
| `lorentz` | float | 0.0 | Manifold regularization weight |
| `beta` | float | 1.0 | KL divergence weight (β-VAE) |
| `dip` | float | 0.0 | DIP-VAE loss weight |
| `tc` | float | 0.0 | Total Correlation loss weight |
| `info` | float | 0.0 | MMD loss weight |
| `loss_type` | str | `'nb'` | `'nb'`, `'zinb'`, `'poisson'`, or `'zip'` |

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr` | float | 1e-4 | Learning rate |
| `batch_size` | int | 128 | Mini-batch size |
| `train_size` | float | 0.7 | Training set proportion |
| `val_size` | float | 0.15 | Validation set proportion |
| `test_size` | float | 0.15 | Test set proportion |

## Methods

### Training
```python
model.fit(epochs=400, patience=25, val_every=5, early_stop=True)
```

### Extraction
```python
latent = model.get_latent()           # Latent representations
bottleneck = model.get_bottleneck()   # Information bottleneck embeddings
pseudotime = model.get_time()         # Pseudotime (ODE mode)
transitions = model.get_transition()  # Transition probabilities (ODE mode)
```

## Citation

If you use Liora in your research, please cite:

```bibtex
@software{liora2025,
  title = {Liora: Lorentz Information ODE-Regularized Variational Autoencoder},
  author = {Zeyu Fu},
  year = {2025},
  url = {https://github.com/PeterPonyu/Liora}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Support

- **Issues**: [GitHub Issues](https://github.com/PeterPonyu/Liora/issues)
- **Email**: fuzeyu99@126.com
- **Documentation**: [GitHub Repository](https://github.com/PeterPonyu/Liora)
