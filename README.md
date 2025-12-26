# Machine Learning Feature Selection — SOTA EDA & Cleaning

A didactic Colab-ready tutorial demonstrating SOTA exploratory data analysis (EDA), cleaning, and feature selection for tabular datasets using Polars, Sweetviz, Pyjanitor, and XGBoost.

This repository contains:

- `tutorial_eda_feature_selection.ipynb` — A progressive Google Colab notebook that walks you through dataset download, EDA, cleaning, and feature selection with explanations and runnable cells.

Why this repo:

- Designed for Kaggle-style tabular problems (100MB–5GB)
- Uses fast, modern tooling: `polars` for performance, `sweetviz` for automated EDA, `pyjanitor` for clean transforms, and `xgboost` for embedded feature selection.

Quick Start (local or Colab)

1. Open the notebook in VS Code or upload it to Google Colab. The notebook includes an "Open in Colab" badge.
2. Run the first cell to install dependencies, then run cells sequentially.

Dependencies

- `polars` — Fast DataFrame library: https://pola.rs/
- `pandas` — Data analysis library: https://pandas.pydata.org/
- `sweetviz` — Automated EDA reports: https://github.com/fbdesignpro/sweetviz
- `pyjanitor` — Clean, chainable data cleaning helpers: https://pyjanitor.readthedocs.io/
- `xgboost` — Gradient boosting library: https://xgboost.ai/
- `scikit-learn` — Classic ML tools and selection methods: https://scikit-learn.org/

(You can install them in Colab with the notebook's first cell.)

Contributing

- Suggest improvements via issues or PRs.

License

This repository is provided as-is for educational purposes. (Add your preferred license.)

---

## About
See `ABOUT.md` for project goals, author, and contact details.