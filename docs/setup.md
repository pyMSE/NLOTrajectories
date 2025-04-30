# Project Setup Guide

Welcome! This guide will help you set up the development environment for this project.

## Requirements

Before you start, make sure you have the following installed:

- Python 3.12  
- Poetry version **2.0.0 or higher**

Check your versions:

```bash
python3 --version
poetry --version
```

If you don’t have Poetry installed or need to upgrade, visit:  
https://python-poetry.org/docs/#installation

## Install dependencies

From the root of the project directory, run:

```bash
poetry install
```

This will create a virtual environment and install all project dependencies.

## Run commands in the virtual environment

Use `poetry run` to execute commands within the virtual environment:

```bash
poetry run python
poetry run pytest
```

If this doesn't work, make sure virtual environments are enabled and that Python 3.12 is used:

```bash
poetry config virtualenvs.create true
poetry env use python3.12
```

## Verify setup

To confirm everything is set up correctly:

```bash
poetry run python --version
```

You should see output like:

```
Python 3.12.x
```

---

You're now ready to start working on the project!