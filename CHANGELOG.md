# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-11-19

### Added
- Transformer-based encoder option with self-attention mechanism
- Multiple ODE function types: legacy, time-conditioned MLP, GRU-based
- Configurable ODE solver parameters (method, step_size, rtol, atol)
- Separate `ode_hidden_dim` parameter to decouple ODE capacity from model hidden dimension
- Comprehensive test suite for ODE configurations
- Full documentation and API reference

### Changed
- Restructured package for PyPI distribution
- Improved ODE solver interface with CPU optimization
- Enhanced manifold regularization with both Lorentz and Euclidean options

### Fixed
- Time encoder dimension mismatch for attention-based encoders
- GRU ODE hidden state management across trajectories
- Import paths and package structure

## [0.2.0] - Earlier

### Added
- Initial VAE implementation with Lorentz manifold regularization
- Information bottleneck architecture
- Neural ODE integration for trajectory inference
- Multiple count-based likelihoods (NB, ZINB, Poisson, ZIP)

## [0.1.0] - Initial Release

### Added
- Basic VAE for single-cell RNA-seq analysis
- Core training loop and data preprocessing

## [0.4.0] - 2025-11-19

### Changed
- Bump version to 0.4.0; align docs and publishing guides
- Update README links (GitHub URLs, contact, citation)

### Fixed
- Publishing guides referencing 0.3.0 now updated to 0.4.0
