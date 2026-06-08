# Personal Portfolio Web App (Flask + live ML model)

A personal portfolio website built with **Flask** that also serves an **interactive
machine-learning model**: visitors fill a form and get a live prediction, alongside CV,
Python projects and contact pages.

## Features
- **Multi-page portfolio**: home, CV, Python projects, contact (Jinja templates + CSS).
- **Live ML prediction page** (`/ml_model`): trains a model at startup from
  `data_to_practice/` and serves predictions from user form input, with categorical
  features handled via `LabelEncoder`.
- **Notebook rendering**: utilities to export/convert notebooks (`nbconvert`, `pdfkit`)
  so analysis notebooks can be shown on the site.
- **Reusable ML utils** (`model_utils.py`): `read_data`, `preprocess_data`, `train_model`.

## Datasets (`data_to_practice/`)
- `data_deposit.csv` — bank-deposit style tabular data (drives the live model)
- `desercion.csv` — customer/student churn (dropout) data; feature-selection notebook in `notebooks/`

## Tech stack
Python · Flask · scikit-learn · pandas · nbconvert / pdfkit · HTML/CSS (Jinja templates)

## Run it
```bash
pip install -r requirements.txt
python app.py        # serves the site; open / and /ml_model
```

## Layout
```
app.py                 Flask routes (home, cv, python-projects, contact, ml_model)
model_utils.py         read_data / preprocess_data / train_model
notebooks/             EDA & feature-selection notebooks
templates/  static/    Jinja templates + CSS
data_to_practice/      sample datasets
```
