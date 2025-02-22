.PHONY: install lint format type-check test

install:
	poetry install --with dev,test,benchmark

lint:
	poetry run ruff check .

develop:
	poetry install --with dev,test,benchmark
	pip install -e .

format:
	poetry run ruff format .
	poetry run ruff check --fix .

type-check:
	poetry run mypy .

test:
	poetry run pytest tests/

benchmark:
	poetry run pytest benchmarks/ --benchmark-only

memory-profile:
	poetry run python -m memory_profiler benchmarks/benchmark_memory.py
