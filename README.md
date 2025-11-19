# Liora

**Lorentz Information ODE-Regularized Variational Autoencoder for single-cell RNA-seq**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Liora is a deep learning framework for single-cell RNA-seq analysis that combines:

- **Variational Autoencoder (VAE)** for dimensionality reduction
- **Lorentz (hyperbolic) or Euclidean manifold regularization** for capturing hierarchical structure
- **Information bottleneck** for hierarchical representation learning
- **Neural ODE dynamics** for continuous trajectory inference (optional)
- **Count-based likelihoods**: Negative Binomial (NB), Zero-Inflated NB (ZINB), Poisson, Zero-Inflated Poisson (ZIP)
- **Flexible encoder architectures**: MLP (default) or Transformer-based with self-attention
- **Multiple ODE function types**: Legacy MLP, time-conditioned MLP, or GRU-based with memory

## Features

### Encoder Options
- **MLP Encoder** (default): Two-layer fully connected network with ReLU activations
- **Transformer Encoder**: Self-attention mechanism with configurable heads, layers, and embedding dimensions
  - Projects features into token sequences
  - Applies multi-head attention across feature space
  - Mean-pools output for latent representation

### ODE Function Types
- **Legacy**: Time-invariant MLP (backward compatibility)
- **Time-conditioned MLP**: Time-aware dynamics with multiple conditioning strategies
  - `concat`: Concatenate time to latent state
  - `film`: Feature-wise Linear Modulation (scale and shift)
  - `add`: Additive time embedding
- **GRU-based**: Recurrent dynamics with trajectory memory for smooth developmental processes

### ODE Solvers
Supports all torchdiffeq methods:
- **Fixed-step**: `rk4`, `euler`, `midpoint` (use `ode_step_size`)
- **Adaptive**: `dopri5`, `adams`, `bosh3` (use `ode_rtol`, `ode_atol`)

## Installation

### From PyPI

```bash
pip install liora
```

### From Source

```bash
git clone https://github.com/PeterPonyu/liora.git
cd liora
pip install -e .
```

### With Test Dependencies

```bash
pip install -e .[test]
```

## Quick Start

```python
import scanpy as sc
from liora import Liora

# Load your data
adata = sc.read_h5ad('data.h5ad')

# Basic usage with default MLP encoder
model = Liora(
    adata,
    layer='counts',
    hidden_dim=128,
    latent_dim=10,
    i_dim=2,
)
model.fit(epochs=100)
latent = model.get_latent()

# Advanced: Transformer encoder + ODE trajectory inference
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
pseudotime = model.get_time()         # Predicted pseudotime (ODE mode)
transitions = model.get_transition()  # Transition matrix (ODE mode)
```

## API Reference

### Main Parameters

#### Model Architecture
- `hidden_dim` (int, default=128): Hidden layer dimension in encoder/decoder
- `latent_dim` (int, default=10): Primary latent space dimensionality
- `i_dim` (int, default=2): Information bottleneck dimension (must be < latent_dim)

#### Encoder Configuration
- `encoder_type` (str, default='mlp'): Encoder architecture
  - `'mlp'`: Standard multi-layer perceptron
  - `'transformer'`: Self-attention based encoder
- `attn_embed_dim` (int, default=64): Embedding dimension for transformer tokens
- `attn_num_heads` (int, default=4): Number of attention heads
- `attn_num_layers` (int, default=2): Number of transformer encoder layers
- `attn_seq_len` (int, default=32): Number of token sequences

#### ODE Configuration
- `use_ode` (bool, default=False): Enable Neural ODE regularization
- `ode_type` (str, default='time_mlp'): ODE function architecture
  - `'legacy'`: Time-invariant MLP (not recommended)
  - `'time_mlp'`: Time-conditioned MLP (recommended)
  - `'gru'`: GRU-based with recurrent memory
- `ode_time_cond` (str, default='concat'): Time conditioning for 'time_mlp'
  - `'concat'`: Concatenate time to state
  - `'film'`: Feature-wise Linear Modulation
  - `'add'`: Additive time embedding
- `ode_hidden_dim` (int, optional): Hidden dimension for ODE networks (defaults to `hidden_dim`)
- `ode_solver_method` (str, default='rk4'): Torchdiffeq solver method
- `ode_step_size` (float or 'auto', optional): Step size for fixed-step solvers
- `ode_rtol` (float, optional): Relative tolerance for adaptive solvers
- `ode_atol` (float, optional): Absolute tolerance for adaptive solvers

#### Loss Configuration
- `recon` (float, default=1.0): Reconstruction loss weight
- `irecon` (float, default=0.0): Information bottleneck reconstruction weight
- `lorentz` (float, default=0.0): Manifold regularization weight
- `beta` (float, default=1.0): KL divergence weight (β-VAE)
- `dip` (float, default=0.0): DIP-VAE loss weight
- `tc` (float, default=0.0): Total Correlation loss weight
- `info` (float, default=0.0): MMD loss weight
- `loss_type` (str, default='nb'): Count likelihood model
  - `'nb'`: Negative Binomial (recommended for UMI data)
  - `'zinb'`: Zero-Inflated Negative Binomial
  - `'poisson'`: Poisson
  - `'zip'`: Zero-Inflated Poisson

#### Training Configuration
- `lr` (float, default=1e-4): Learning rate for Adam optimizer
- `batch_size` (int, default=128): Mini-batch size
- `train_size` (float, default=0.7): Proportion of training data
- `val_size` (float, default=0.15): Proportion of validation data
- `test_size` (float, default=0.15): Proportion of test data

### Methods

- `fit(epochs=400, patience=25, val_every=5, early_stop=True)`: Train the model
- `get_latent()`: Extract latent representations
- `get_bottleneck()`: Extract information bottleneck embeddings
- `get_time()`: Get predicted pseudotime (ODE mode only)
- `get_transition()`: Get cell-to-cell transition probabilities (ODE mode only)

## Testing

Run the test suite:

```bash
pytest
```

Or with coverage:

```bash
pytest --cov=liora --cov-report=html
```

Tests cover:
- ODE function construction for all types
- Fixed-step and adaptive ODE solvers
- VAE forward passes with different configurations
- Encoder variants (MLP and Transformer)

## Examples

See the `examples/` directory for:
- `encoder_example.py`: Demonstrates MLP vs Transformer encoders
- More examples coming soon!

## Technical Notes

### ODE Solver Selection

**Fixed-step solvers** (rk4, euler, midpoint):
- Require `ode_step_size` parameter
- Use `'auto'` for uniform discretization based on time points
- Good for when trajectory is well-behaved

**Adaptive solvers** (dopri5, adams):
- Automatically adjust step size based on error tolerance
- Use `ode_rtol` and `ode_atol` for control
- Better for stiff or complex dynamics

### GRU ODE Notes

The GRU-based ODE maintains hidden state across time steps:
- Hidden state automatically resets before each trajectory
- Provides smooth dynamics with memory of past states
- Best for modeling developmental processes

### CPU vs GPU for ODE

ODE solving runs on CPU for computational efficiency:
- torchdiffeq is CPU-optimized for small latent dimensions
- Observed 2-3x speedup vs GPU for typical use cases
- Avoids GPU memory pressure

## Citation

If you use Liora in your research, please cite:

```bibtex
@software{liora2025,
  title = {Liora: Lorentz Information ODE-Regularized Variational Autoencoder},
  author = {Zeyu Fu},
  year = {2025},
  url = {https://github.com/PeterPonyu/liora}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

- Issues: https://github.com/PeterPonyu/liora/issues
- Email: fuzeyu99@126.com
