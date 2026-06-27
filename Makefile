.PHONY: install install-dev run demo test lint format models

install:
	python3 -m pip install -e .

install-dev:
	python3 -m pip install -e ".[dev]"

run:
	python3 -m streamlit run src/voyager_ai/app.py

demo:
	VOYAGER_LLM_PROVIDER=demo python3 -m streamlit run src/voyager_ai/app.py

test:
	PYTHONPATH=src python3 -m unittest discover -s tests

lint:
	ruff check src tests

format:
	ruff format src tests

models:
	PYTHONPATH=src python3 -m voyager_ai models
