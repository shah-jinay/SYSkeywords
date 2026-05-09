.PHONY: setup test lint clean ingest

setup:
	pip install -r requirements-dev.txt

test:
	PYTHONPATH=src pytest tests/ --cov=tailor --cov-report=term-missing -v

lint:
	PYTHONPATH=src python -m py_compile src/tailor/*.py && echo "Syntax OK"

ingest:
	PYTHONPATH=src python -m tailor.cli ingest

clean:
	rm -f data/db/library.sqlite data/db/history.sqlite
	rm -rf out/*.pdf
	rm -rf src/tailor/__pycache__ tests/__pycache__
	rm -rf .pytest_cache
