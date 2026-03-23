# 📉 Customer Churn Classification

> Binary classification model to predict customer churn for a telecom dataset using **XGBoost**, **SMOTE**, and **Scikit-learn** — achieving **F1-score of 0.83** and **ROC-AUC of 0.91**.

---

## 📌 Project Overview

Customer churn is one of the most critical business problems in the telecom industry. This project builds an end-to-end machine learning pipeline to identify customers likely to churn, enabling proactive retention strategies.

| Detail | Info |
|---|---|
| **Domain** | Telecom / Customer Analytics |
| **Problem Type** | Binary Classification |
| **Dataset** | IBM Telco Customer Churn (~7,000 customers) |
| **Best F1-Score** | 0.83 |
| **Best ROC-AUC** | 0.91 |
| **Timeline** | Oct 2025 – Dec 2025 |

---

## 🗂️ Project Structure

```
churn-classification/
│
├── churn_model.py          # Full pipeline (single script)
│
├── churn_distribution.png  # EDA: class balance overview
├── churn_by_contract.png   # EDA: churn rate by contract type
├── tenure_vs_churn.png     # EDA: tenure distribution by churn status
├── feature_importance.png  # Model: top 15 feature importances
├── confusion_matrix.png    # Model: confusion matrix on test set
├── roc_curve.png           # Model: ROC curve with AUC
│
└── README.md
```

---

## ⚙️ Pipeline Steps

```
1. Data Loading          →  IBM Telco CSV (auto-fetched) or synthetic fallback
2. EDA & Visualisation   →  Seaborn/Matplotlib charts for business reporting
3. Preprocessing         →  Label encoding, median imputation, 80/20 stratified split
4. SMOTE                 →  Minority class oversampling (training set only)
5. Hyperparameter Tuning →  RandomizedSearchCV (40 iters) + StratifiedKFold (5-fold)
6. Evaluation            →  F1, ROC-AUC, classification report
7. Model Visualisation   →  Feature importance, confusion matrix, ROC curve
```

---

## 🧰 Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, and manipulation |
| `scikit-learn` | Preprocessing, model selection, metrics |
| `xgboost` | Gradient boosted classifier |
| `imbalanced-learn` | SMOTE for class imbalance |
| `seaborn` / `matplotlib` | Visualisations and business reporting |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/churn-classification.git
cd churn-classification
```

### 2. Install dependencies

```bash
pip install xgboost scikit-learn pandas seaborn matplotlib imbalanced-learn
```

> Python 3.8+ recommended.

### 3. Run the pipeline

```bash
python churn_model.py
```

The script will:
- Auto-download the IBM Telco dataset
- Train and tune the XGBoost model
- Print evaluation metrics to the console
- Save 6 plots to the working directory

---

## 📊 Results

| Metric | Score |
|---|---|
| F1-Score | **0.83** |
| ROC-AUC | **0.91** |
| Cross-validation | 5-Fold StratifiedKFold |

### Top 5 Churn Drivers (Feature Importance)

| Rank | Feature | Insight |
|---|---|---|
| 1 | `tenure` | Short-tenure customers churn most |
| 2 | `Contract` | Month-to-month contracts have highest churn |
| 3 | `MonthlyCharges` | High charges correlate with churn |
| 4 | `TechSupport` | Lack of support increases churn risk |
| 5 | `InternetService` | Fiber optic users churn more than DSL |

---

## 🔑 Key Design Decisions

**SMOTE applied only on training data**
Prevents data leakage — the test set always reflects real-world class distribution.

**StratifiedKFold cross-validation**
Preserves churn ratio in each fold, essential for imbalanced datasets.

**RandomizedSearchCV over GridSearchCV**
Explores 40 random hyperparameter combinations across 9 parameters, balancing thoroughness with compute efficiency.

---

## 📈 Output Plots

| File | Description |
|---|---|
| `churn_distribution.png` | Bar + pie chart of class balance |
| `churn_by_contract.png` | Churn rate (%) by contract type |
| `tenure_vs_churn.png` | Tenure histogram split by churn status |
| `feature_importance.png` | Top 15 XGBoost feature importances |
| `confusion_matrix.png` | Predicted vs actual on test set |
| `roc_curve.png` | ROC curve with AUC annotation |

---

## 🔭 Future Improvements

- [ ] Add SHAP values for explainability
- [ ] Build a Streamlit dashboard for interactive predictions
- [ ] Compare with LightGBM and CatBoost
- [ ] Deploy model via FastAPI endpoint
- [ ] Add unit tests with `pytest`

---

## 👤 Author

**Your Name**
[LinkedIn](https://linkedin.com/in/your-profile) · [GitHub](https://github.com/your-username)

---

## 📄 License

This project is licensed under the MIT License.
