# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2025-12-26

### Changed
- Renamed package from "liora" to "laior" to fully align with LAIOR naming
- Updated all imports, references, and documentation to use "laior" package name
- Maintained backward compatibility aliases where possible
- Updated version to 0.6.0 for new release

## [0.5.0] - 2025-12-25

### Changed
- Renamed model from "Liora" to "LAIOR" (Lorentz Attentive Interpretable ODE Regularized VAE) to align with published paper
- Updated full name from "Lorentz Interpretable ODE-Regularized Attention-based VAE" to match paper specification
- Maintained backward compatibility: `Liora` class name remains as alias to `LAIOR`
- Updated all documentation, docstrings, and comments to reflect LAIOR naming
- Enhanced description to emphasize cross-modality support (scRNA-seq and scATAC-seq)

## [0.4.2] - 2025-11-21

### Changed
- Streamlined README.md with table-based configuration guide
- Removed Testing, Examples, and Technical Notes sections for cleaner documentation
- Added PyPI badge to README
- Improved overall documentation clarity and professional presentation

## [0.4.1] - 2025-11-19

### Fixed
- Python 3.9 compatibility: Replaced `|` union type syntax with `Optional[]` from typing
- Added missing `typing.Optional` imports to agent.py and model.py
- CI/CD tests now pass on Python 3.9

## [0.4.0] - 2025-11-19

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
- Updated README links (GitHub URLs, contact, citation)

### Fixed
- Time encoder dimension mismatch for attention-based encoders
- GRU ODE hidden state management across trajectories
- Import paths and package structure

## [0.3.0] - 2025-11-19

### Changed
- Initial PyPI-ready package structure

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
