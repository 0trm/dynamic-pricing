.PHONY: data train evaluate sanity test app all clean

PYTHON ?= python

data:
	$(PYTHON) -m src.data.make_dataset

train:
	$(PYTHON) -m src.models.train_demand_model

evaluate:
	$(PYTHON) -m src.models.evaluate

sanity:
	$(PYTHON) -m src.models.sanity_check

test:
	$(PYTHON) -m pytest -q tests

app:
	streamlit run app.py

all: data train evaluate sanity

clean:
	rm -f models/*.joblib models/*.json
	rm -f proposals.jsonl
