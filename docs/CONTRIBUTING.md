# Contributing to Thermal Homography

Thank you for your interest in contributing to this research project!

## Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/YaqoobAnsari/thermal-homography-equivariant.git
   cd thermal-homography-equivariant
   ```

2. **Create a conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate thermal-homography
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

## Code Style

We use the following tools to maintain code quality:

- **Black** for code formatting (line length: 100)
- **Ruff** for linting
- **MyPy** for type checking

Run all checks:
```bash
black src/
ruff check src/
mypy src/ --ignore-missing-imports
```

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run setup verification:
```bash
python scripts/verify_setup.py
```

## Pull Request Process

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

3. Push to your fork and create a Pull Request

4. Ensure all CI checks pass

## Commit Message Format

We follow conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

## Reporting Issues

- Use the issue templates provided
- Include full error tracebacks
- Include your environment details (OS, Python version, etc.)

## Code of Conduct

Be respectful and constructive in all interactions.
