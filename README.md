<div align="center">

# 🧪 DS Playbook

### A Comprehensive Data Science Toolkit & Portfolio

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![CI](https://github.com/adityapathak2004/project-1/actions/workflows/ci.yml/badge.svg?style=for-the-badge)](https://github.com/adityapathak2004/project-1/actions/workflows/ci.yml)
[![Jupyter](https://img.shields.io/badge/Notebooks-4-orange?style=for-the-badge&logo=jupyter&logoColor=white)](notebooks/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red?style=for-the-badge&logo=streamlit&logoColor=white)](app/)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-black?style=for-the-badge)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen?style=for-the-badge)](https://github.com/adityapathak2004/project-1/pulls)

**Battle-tested data science toolkit: auto-EDA, feature engineering, model benchmarking, and an interactive Streamlit dashboard — all in clean, reusable Python.**

[Features](#-features) • [Quick Start](#-quick-start) • [Notebooks](#-notebooks) • [Dashboard](#-dashboard) • [Tech Stack](#️-tech-stack)

</div>

---

## 📦 What's Inside

| Module | Description |
|--------|-------------|
| `dskit.eda` | Auto EDA — missing values, outliers, distributions, correlations in **one function call** |
| `dskit.features` | Feature engineering — smart encoding, date extraction, scaling, imputation |
| `dskit.models` | Model shootout — train & benchmark 5+ models simultaneously with cross-validation |
| `dskit.viz` | Publication-ready charts — Matplotlib wrappers with sensible defaults |
| `app/dashboard.py` | Interactive Streamlit dashboard — upload any CSV → instant analysis in the browser |

---

## ✨ Features

- **🔍 Auto-EDA** — one call gives you dtypes, nulls, outliers, correlations, and distributions
- **⚙️ Smart Feature Engineering** — auto-detects column types and applies the right transforms
- **🤖 Model Shootout** — Logistic Regression, Random Forest, GBM, SVM side-by-side with CV
- **📈 Time Series** — decomposition, stationarity tests, ARIMA forecasting
- **📊 Beautiful Vizzes** — publication-ready Matplotlib/Seaborn figures
- **🖥️ Live Dashboard** — upload CSV → full analysis without writing a line of code
- **✅ CI Tested** — GitHub Actions runs pytest on Python 3.9, 3.10, 3.11 on every push
- **📓 4 Notebooks** — end-to-end walkthroughs from raw data to deployed insights

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/adityapathak2004/project-1.git
cd project-1

# Virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install
pip install -r requirements.txt
pip install -e .

# Launch dashboard
streamlit run app/dashboard.py

# Run tests
pytest tests/ -v
```

---

## 🎯 Usage

```python
import pandas as pd
from dskit.eda import auto_eda, missing_report
from dskit.features import encode_categoricals, scale_features, handle_missing
from dskit.models import compare_models
from dskit.viz import plot_correlation_matrix, plot_model_comparison

df = pd.read_csv('your_data.csv')

# Full EDA in one call
report = auto_eda(df, target='target_col')
print(report['missing'])
print(report['target_analysis'])

# Preprocess
df = handle_missing(df, strategy='median')
df = encode_categoricals(df)
df = scale_features(df)

# Compare models with cross-validation
X, y = df.drop(columns=['target_col']), df['target_col']
results = compare_models(X, y, task='classification', cv=5)
print(results)

# Visualise
plot_correlation_matrix(df)
plot_model_comparison(results)
```

---

## 📓 Notebooks

| # | Notebook | Topics Covered |
|---|----------|-----------------|
| 01 | [EDA Masterclass](notebooks/01_eda_masterclass.ipynb) | Auto-EDA, distributions, outliers, correlations, target analysis |
| 02 | [Feature Engineering](notebooks/02_feature_engineering.ipynb) | Encoding, scaling, date features, imputation, polynomial features |
| 03 | [ML Model Comparison](notebooks/03_ml_model_comparison.ipynb) | CV benchmark, model ranking, feature importance |
| 04 | [Time Series Analysis](notebooks/04_time_series_analysis.ipynb) | Decomposition, ADF test, ACF/PACF, ARIMA forecasting |

---

## 🖥️ Dashboard

```bash
streamlit run app/dashboard.py
```

| Tab | What you get |
|-----|--------------|
| **📊 Overview** | Row/column count, data types, preview |
| **🔍 EDA** | Missing values, numeric stats, distribution plots |
| **🔗 Correlations** | Heatmap, top correlated pairs |
| **🤖 Models** | One-click model benchmark with CV + performance charts |

---

## 📁 Project Structure

```
project-1/
├── .github/workflows/ci.yml  # GitHub Actions CI
├── src/dskit/
│   ├── eda.py          # Auto-EDA utilities
│   ├── features.py     # Feature engineering
│   ├── models.py       # Model comparison framework
│   └── viz.py          # Visualization helpers
├── notebooks/
│   ├── 01_eda_masterclass.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_ml_model_comparison.ipynb
│   └── 04_time_series_analysis.ipynb
├── app/dashboard.py    # Streamlit dashboard
├── tests/              # pytest suite (22 tests)
├── requirements.txt
└── setup.py
```

---

## 🛠️ Tech Stack

| Category | Libraries |
|----------|-----------|
| Core | Python 3.9+, NumPy, Pandas, SciPy |
| ML | Scikit-learn, XGBoost, LightGBM |
| Visualization | Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Time Series | statsmodels |
| Testing | pytest, pytest-cov |
| CI | GitHub Actions |

---

## 📄 License

MIT — use it, learn from it, ship it.

---

<div align="center">

Built with ☕ and way too many hours in Jupyter notebooks.

**[⭐ Star this repo](https://github.com/adityapathak2004/project-1)** if it helps your data science journey!

</div>
