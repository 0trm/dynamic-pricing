.PHONY: data train evaluate sanity test app all clean figures

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

# Paper charts, PDF and README figures (needs Typst: https://typst.app)
figures:
	$(PYTHON) paper/figures.py
	typst compile --root . paper/dynamic-pricing.typ
	for f in context architecture panel; do typst compile --root . --input fig=$$f --format svg paper/readme-figures.typ assets/$$f.svg; done

clean:
	rm -f models/*.joblib models/*.json
	rm -f proposals.jsonl
