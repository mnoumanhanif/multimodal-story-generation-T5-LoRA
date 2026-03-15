# Contributing to Multimodal Story Generation

Thank you for your interest in contributing! This document provides
guidelines and instructions for contributing to this project.

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:

   ```bash
   git clone https://github.com/<your-username>/multimodal-story-generation-T5-LoRA.git
   cd multimodal-story-generation-T5-LoRA
   ```

3. **Create a virtual environment** and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux / macOS
   pip install -e ".[dev]"
   ```

4. **Create a branch** for your changes:

   ```bash
   git checkout -b feature/my-feature
   ```

## Development Workflow

### Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions.
- Maximum line length: **100 characters**.
- Use descriptive variable and function names.
- Add docstrings to all public functions and classes.

### Linting

```bash
flake8 src/ tests/ --max-line-length=100
```

### Running Tests

```bash
pytest tests/ -v
```

### Commit Messages

- Use clear, concise commit messages.
- Start with a verb in the imperative mood (e.g., "Add", "Fix", "Update").
- Reference issue numbers where applicable (e.g., "Fix #42").

## Pull Request Process

1. Ensure your changes pass linting and all tests.
2. Update documentation if your changes affect public APIs.
3. Open a pull request against the `main` branch.
4. Fill in the pull request template.
5. Wait for a maintainer review before merging.

## Reporting Issues

- Use the [issue tracker](https://github.com/mnoumanhanif/multimodal-story-generation-T5-LoRA/issues).
- Include steps to reproduce, expected behaviour, and actual behaviour.
- Attach error logs or screenshots where helpful.

## Code of Conduct

Be respectful and constructive in all interactions. We follow the
[Contributor Covenant](https://www.contributor-covenant.org/).

## License

By contributing, you agree that your contributions will be licensed
under the MIT License.
