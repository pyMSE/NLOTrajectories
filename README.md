# NLOTrajectories
Robot Motion Planning: Trajectory optimization in learned environments

This project contains benchmark tasks for trajectory optimization using CasADi and l4casadi.

## Benchmarks
- **Benchmark 1**: Point mass + circular obstacle + approximated SDF (mode: casadi) 
`src/nlotrajectories/benchmarks/benchmark_1_dot_circle.yaml`
- **Benchmark 2**: Rectangular unicycle + circular obstacle + approximated SDF (mode: casadi)
`src/nlotrajectories/benchmarks/benchmark_2_unicycle_circle.yaml`
- **Benchmark 3**: Rectangular unicycle + multiple convex obstacles + learned SDF (mode: l4casadi)
`src/nlotrajectories/benchmarks/benchmark_3_unicycle_convex.yaml`
- **Benchmark 4**: Rectangular unicycle avoiding a star-shaped obstacle
`src/nlotrajectories/benchmarks/benchmark_4_dot_nonconvex.yaml`
- **Benchmark 5**: Car (Ackermann) dynamics + circular obstacle
`src/nlotrajectories/benchmarks/benchmark_5_ackermann_circle.yaml`
- **Benchmark 6**: Car (Ackermann) dynamics + non-convex wave-like corridor obstacle
`src/nlotrajectories/benchmarks/benchmark_6_ackermann_wave.yaml`

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


