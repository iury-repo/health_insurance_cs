# Health Insurance Cross Sell

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A health insurance model to get information about client interebuy a new insurance product

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         insurance_classifier and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── insurance_classifier   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes insurance_classifier a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```
## Notebook Name Convention

```
Notebook name example: 0.01-mni-short-description.ipynb

    0 - Data wrangling / description - often includes cleaning, feature creation and initial diagnostics about the raw dataset. Writes data to data/processed or data/interim
    1 - Data exploration - often just for exploratory work and initial diagnostic about the raw dataset.
    2 - Modeling - training machine learning models
    3 - Visualizations - often writes publication-ready viz to reports
    4 - Publication - Notebooks that get turned directly into reports

* mni - Your initials (My Name Initials); this is helpful for knowing who created the notebook and prevents collisions from people working in the same notebook.
* short-description - A description of what the notebook covers, e.g. (data-cleaning, visualizations, fine-tuning, etc)
```
--------

