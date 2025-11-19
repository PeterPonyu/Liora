# Contributing to Liora

Thank you for your interest in contributing to Liora!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/zeyufu/liora.git
cd liora
```

2. Install in editable mode with dev dependencies:
```bash
pip install -e .[dev]
```

3. Run tests:
```bash
pytest
```

## Code Style

- We use Black for code formatting (line length: 100)
- We use isort for import sorting
- Run formatters before committing:
```bash
black liora/
isort liora/
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and formatters
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to your fork (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Testing

- Write tests for new features
- Ensure all tests pass before submitting PR
- Aim for high test coverage

## Questions?

Open an issue for discussion!
