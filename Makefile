.PHONY: setup-data data notebooks test all clean

HISTORICAL_REPO ?= ../../2017/dynet-german-ABSA-for-github

setup-data:
	mkdir -p data/raw data/canonical data/derived/task_c data/derived/task_d data/manifests
	ln -sf $(abspath $(HISTORICAL_REPO))/data/train_v1.4.xml data/raw/train_v1.4.xml
	ln -sf $(abspath $(HISTORICAL_REPO))/data/dev_v1.4.xml data/raw/dev_v1.4.xml
	ln -sf $(abspath $(HISTORICAL_REPO))/data/test_TIMESTAMP1.xml data/raw/test_TIMESTAMP1.xml
	ln -sf $(abspath $(HISTORICAL_REPO))/data/test_TIMESTAMP2.xml data/raw/test_TIMESTAMP2.xml
	ln -sf $(abspath $(HISTORICAL_REPO))/test_TIMESTAMP1.tsv data/raw/test_TIMESTAMP1.tsv
	ln -sf $(abspath $(HISTORICAL_REPO))/data/test_TIMESTAMP2.tsv data/raw/test_TIMESTAMP2.tsv
	ln -sf $(abspath $(HISTORICAL_REPO))/GermEval2017/EvaluationScript.jar data/raw/EvaluationScript.jar

data: setup-data
	uv run python -m germeval_llm.pipeline

notebooks:
	uv run python -m germeval_llm.build_notebooks

test:
	uv run pytest -v

all: data notebooks test
