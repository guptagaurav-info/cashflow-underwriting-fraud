PY ?= python
export PYTHONPATH := src

.PHONY: all ingest simulate test clean
all:
	$(PY) -m cfu.pipeline
ingest:
	$(PY) -m cfu.ingest
simulate:
	$(PY) -m cfu.simulate
test:
	$(PY) -m pytest -q tests
clean:
	rm -rf data/processed/* reports/figures/*.png reports/memos/* reports/metrics.json
