# Justfile for linting, formatting, and testing

# Default recipe to show available commands
default:
    @just --list

# Python linting and formatting with Ruff
lint-python:
    uv run poe lint

fix-python:
    uv run poe fix

format-python:
    uv run poe format

# Python type checking with MyPy
type-check:
    uv run poe type-check

# YAML linting
lint-yaml:
    uv run poe lint-yaml

# Testing with pytest
test:
    uv run poe test

# Test with coverage
test-cov:
    uv run poe test-cov

# Generate coverage report
coverage-report:
    uv run poe coverage-report

# JSON linting with Ruff (supports JSON)
lint-json:
    uv run ruff check --select=JSON *.json **/*.json

format-json:
    uv run ruff format *.json **/*.json

# Rust linting and formatting
lint-rust:
    cargo clippy -- -D warnings

fix-rust:
    cargo clippy --fix --allow-dirty

format-rust:
    cargo fmt

check-rust:
    cargo check

# TOML formatting (if you have taplo installed)
lint-toml:
    taplo fmt --check .

format-toml:
    taplo fmt .

# Markdown linting
lint-markdown:
    uv run poe lint-markdown

# Run all linting tasks
lint-all: lint-python lint-yaml lint-json lint-rust lint-toml

# Run all formatting tasks
format-all: format-python format-json format-rust format-toml

# Run all checks (lint + type check)
check-all: lint-all type-check

# Run tests and checks
test-all: test-cov check-all

# Clean up Python cache files
clean:
    find . -type f -name "*.pyc" -delete
    find . -type d -name "__pycache__" -delete
    find . -type d -name "*.egg-info" -exec rm -rf {} +