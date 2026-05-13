.PHONY: data train evaluate sanity app all clean

PYTHON ?= python

data:
	$(PYTHON) -m src.data.make_dataset

train:
	$(PYTHON) -m src.models.train_demand_model

evaluate:
	$(PYTHON) -m src.models.evaluate

sanity:
	$(PYTHON) -m src.models.sanity_check

app:
	streamlit run app.py

all: data train evaluate sanity

clean:
	rm -f models/*.joblib
	rm -f proposals.jsonl
