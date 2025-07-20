# NLOTrajectories
Robot Motion Planning: Trajectory optimization in learned environments

This project contains benchmark tasks for trajectory optimization using CasADi and l4casadi.

## Benchmarks
- **Benchmark 1**: Point mass avoiding a circular obstacle
- **Benchmark 2**: Car (unicycle) navigating convex obstacles
- **Benchmark 3**: Car in maze  (?)
- **Benchmark 4**: Learned dynamics + CasADi (?)
- **Benchmark 5**: Learned dynamics + l4casadi  (?)

## Getting Started

```bash
poetry install
poetry run pip install l4casadi --no-build-isolation
```

## Running a Benchmark

```bash
poetry run run-benchmark --config <BENCHMARK_PATH>
```

Example:

```bash
poetry run run-benchmark --config src/nlotrajectories/benchmarks/benchmark_1_dot_circle.yaml
```

## Running Tests

```bash
poetry run pytest
```

## Code Style

Before pushing any changes, make sure code formatting and linting passes:

```bash
poetry run make pretty lint
```


## Frontend

Interactive widget for designing benchmark scenarios with point mass robot and convex obstacles: 

```bash
poetry run python src/nlotrajectories/scripts/tkinter_editor.py
```

![Alt text](docs/frontend.png?raw=true "Title")


