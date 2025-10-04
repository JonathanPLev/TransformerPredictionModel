# TransformerPredictionModel

Project to predict sports player performance outcomes using complex ML modeling.

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- [just](https://github.com/casey/just) (command runner)
- [Rust/Cargo](https://www.rust-lang.org/) (for Rust components)
- Git

**Install prerequisites on macOS:**

**Option 1: Homebrew (recommended)**
```bash
brew install uv just rust
```

**Option 2: Official installers**
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install just and rust via Homebrew
brew install just rust
```

**Option 3: Using Rust/Cargo (alternative for just)**
```bash
# Install uv via official installer
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Rust toolchain (includes cargo)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install just via cargo
cargo install just
```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/JonathanPLev/TransformerPredictionModel.git
   cd TransformerPredictionModel
   ```

2. **Set up Python environment**
   ```bash
   # Create virtual environment and install dependencies
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv sync --dev
   ```

3. **Verify installation**
   ```bash
   # Check available commands
   just --list
   
   # Run linting and tests
   just check-all
   ```

## 🛠️ Development

### Available Commands

Use `just <command>` to run these tasks:

#### **Python Development**
- `just lint-python` - Lint Python code with Ruff
- `just fix-python` - Auto-fix Python issues
- `just format-python` - Format Python code
- `just type-check` - Run MyPy type checking

#### **Testing**
- `just test` - Run pytest tests
- `just test-cov` - Run tests with coverage
- `just coverage-report` - Generate HTML coverage report

#### **Other Languages**
- `just lint-yaml` - Lint YAML files
- `just lint-json` - Lint JSON files
- `just lint-rust` - Lint Rust code (if applicable)
- `just format-rust` - Format Rust code

#### **Convenience Commands**
- `just lint-all` - Run all linting tasks
- `just format-all` - Run all formatting tasks
- `just check-all` - Run all checks (lint + type check)
- `just test-all` - Run tests + all checks
- `just clean` - Clean Python cache files

### Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Make your changes and test**
   ```bash
   # Sync dependencies if you added any
   uv sync --dev
   
   # Run checks locally
   just check-all
   just test-cov
   ```

3. **Commit with conventional format**
   ```bash
   git add .
   git commit -m "feat: add new prediction model"
   ```

4. **Push and create PR**
   ```bash
   git push origin feat/your-feature-name
   # Create PR on GitHub
   ```

### Commit Convention

This project follows [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `style:` - Code style changes
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `test:` - Test additions/modifications
- `chore:` - Maintenance tasks

## 🏗️ Project Structure

```
TransformerPredictionModel/
├── .github/                 # GitHub workflows and templates
│   ├── workflows/          # CI/CD workflows
│   └── pull_request_template.md
├── src/                    # Source code (create as needed)
├── tests/                  # Test files (create as needed)
├── pyproject.toml         # Python project configuration
├── justfile               # Task runner commands
├── uv.lock               # Dependency lock file
└── README.md             # This file
```

## 🧪 Testing

Run tests with coverage:
```bash
just test-cov
```

Generate HTML coverage report:
```bash
just coverage-report
# Open htmlcov/index.html in your browser
```

## 📋 Contributing

1. **Fork the repository**
2. **Create a feature branch** following the naming convention: `feat/description`, `fix/description`, etc.
3. **Make your changes** and ensure all checks pass locally
4. **Write tests** for new functionality
5. **Submit a pull request** with a clear description

### PR Requirements
- All CI checks must pass
- Code owner approval required
- Conventional commit format in PR title
- Fill out the PR template completely

## 🔧 Configuration

### Python Dependencies
- **Runtime**: Defined in `pyproject.toml` under `dependencies`
- **Development**: Defined in `pyproject.toml` under `tool.uv.dev-dependencies`

### Code Quality Tools
- **Linting**: Ruff
- **Type Checking**: MyPy
- **Testing**: Pytest with coverage
- **YAML Linting**: yamllint

## 🤝 Support

For questions or issues, please:
1. Check existing [Issues](https://github.com/JonathanPLev/TransformerPredictionModel/issues)
2. Create a new issue with detailed information
3. Follow the issue template guidelines