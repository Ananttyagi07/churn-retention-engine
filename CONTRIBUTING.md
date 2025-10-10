# Contributing to Customer Churn & Retention Engine

We welcome contributions! Please follow these guidelines:

## Development Setup

1. Fork the repository
2. Clone your fork: `git clone https://github.com/Faraazz05/churn-retention-engine.git`
3. Create a virtual environment: `python -m venv venv`
4. Install dependencies: `pip install -r requirements-dev.txt`
5. Install pre-commit hooks: `pre-commit install`

## Making Changes

1. Create a feature branch: `git checkout -b feature/amazing-feature`
2. Make your changes
3. Run tests: `make test`
4. Run linters: `make lint`
5. Format code: `make format`
6. Commit with clear messages: `git commit -m "Add amazing feature"`

## Pull Request Process

1. Update documentation if needed
2. Add tests for new functionality
3. Ensure all tests pass
4. Update CHANGELOG.md
5. Request review from maintainers

## Code Standards

- Follow PEP 8 style guide
- Use Black for formatting (line length 88)
- Add type hints
- Write docstrings (Google style)
- Maintain 80%+ test coverage

## Questions?

Open a GitHub Discussion or reach out via Issues.
