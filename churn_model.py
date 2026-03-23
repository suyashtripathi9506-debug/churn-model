# =============================================================================
# Customer Churn Classification | Telecom Dataset
# Tools: XGBoost, Scikit-learn, Pandas, Seaborn, imbalanced-learn (SMOTE)
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve
)
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ── Plotting style ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
COLORS = {"churn": "#e74c3c", "no_churn": "#2ecc71", "primary": "#2c3e50"}

# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_data() -> pd.DataFrame:
    """
    Load the IBM Telco Customer Churn dataset.
    Falls back to a synthetic dataset if the URL is unreachable.
    """
    url = (
        "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d"
        "/master/data/Telco-Customer-Churn.csv"
    )
    try:
        df = pd.read_csv(url)
        print(f"[✓] IBM Telco dataset loaded  →  {df.shape[0]:,} rows, {df.shape[1]} columns")
    except Exception:
        print("[!] Remote fetch failed — generating synthetic dataset …")
        df = _make_synthetic_dataset(n=7_000)
    return df


def _make_synthetic_dataset(n: int = 7_000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic telecom churn dataset that mirrors IBM Telco structure."""
    rng = np.random.default_rng(seed)

    tenure        = rng.integers(0, 72, n)
    monthly_charges = rng.uniform(18, 120, n).round(2)
    total_charges   = (tenure * monthly_charges + rng.normal(0, 50, n)).clip(0).round(2)

    contract       = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    internet       = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    payment        = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        n, p=[0.34, 0.23, 0.22, 0.21]
    )
    tech_support   = rng.choice(["Yes", "No", "No internet service"], n)
    online_security= rng.choice(["Yes", "No", "No internet service"], n)
    senior         = rng.choice([0, 1], n, p=[0.84, 0.16])
    partner        = rng.choice(["Yes", "No"], n)
    dependents     = rng.choice(["Yes", "No"], n)
    paperless      = rng.choice(["Yes", "No"], n, p=[0.59, 0.41])
    phone_service  = rng.choice(["Yes", "No"], n, p=[0.90, 0.10])
    multiple_lines = rng.choice(["Yes", "No", "No phone service"], n)
    gender         = rng.choice(["Male", "Female"], n)

    # Churn probability influenced by key drivers
    churn_prob = (
        0.05
        + 0.30 * (contract == "Month-to-month")
        + 0.15 * (internet == "Fiber optic")
        + 0.10 * (tech_support == "No")
        + 0.08 * (online_security == "No")
        + 0.07 * (paperless == "Yes")
        - 0.20 * (tenure > 36)
        - 0.10 * (partner == "Yes")
    ).clip(0.02, 0.95)

    churn = (rng.random(n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customerID":       [f"CUST-{i:05d}" for i in range(n)],
        "gender":           gender,
        "SeniorCitizen":    senior,
        "Partner":          partner,
        "Dependents":       dependents,
        "tenure":           tenure,
        "PhoneService":     phone_service,
        "MultipleLines":    multiple_lines,
        "InternetService":  internet,
        "OnlineSecurity":   online_security,
        "TechSupport":      tech_support,
        "Contract":         contract,
        "PaperlessBilling": paperless,
        "PaymentMethod":    payment,
        "MonthlyCharges":   monthly_charges,
        "TotalCharges":     total_charges.astype(str),
        "Churn":            ["Yes" if c else "No" for c in churn],
    })
    return df


# =============================================================================
# 2. PREPROCESSING
# =============================================================================

def preprocess(df: pd.DataFrame):
    """Clean, encode, and split the dataset."""
    df = df.copy()

    # Drop customer ID (non-informative)
    df.drop(columns=["customerID"], errors="ignore", inplace=True)

    # TotalCharges can contain spaces in the IBM dataset
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Target encoding
    df["Churn"] = (df["Churn"].str.strip() == "Yes").astype(int)

    # Separate features / target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Encode all object columns with LabelEncoder
    label_encoders = {}
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    feature_names = X.columns.tolist()
    X_arr = X.values.astype(float)
    y_arr = y.values

    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=0.20, random_state=42, stratify=y_arr
    )

    print(f"[✓] Train size : {X_train.shape[0]:,}  |  Test size : {X_test.shape[0]:,}")
    print(f"[✓] Churn rate : {y_arr.mean()*100:.1f}%  (class imbalance present — SMOTE applied)")

    return X_train, X_test, y_train, y_test, feature_names


# =============================================================================
# 3. CLASS IMBALANCE — SMOTE
# =============================================================================

def apply_smote(X_train, y_train):
    """Oversample the minority class using SMOTE."""
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"[✓] After SMOTE → Churn: {y_res.sum():,}  |  No-churn: {(y_res==0).sum():,}")
    return X_res, y_res


# =============================================================================
# 4. HYPERPARAMETER TUNING — RandomizedSearchCV
# =============================================================================

def tune_model(X_train, y_train) -> XGBClassifier:
    """
    Tune XGBoost hyperparameters using RandomizedSearchCV
    with 5-fold StratifiedKFold cross-validation.
    """
    param_dist = {
        "n_estimators":      [100, 200, 300, 400, 500],
        "max_depth":         [3, 4, 5, 6, 7],
        "learning_rate":     [0.01, 0.05, 0.1, 0.15, 0.2],
        "subsample":         [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree":  [0.5, 0.6, 0.7, 0.8, 1.0],
        "min_child_weight":  [1, 3, 5, 7],
        "gamma":             [0, 0.1, 0.2, 0.3],
        "reg_alpha":         [0, 0.01, 0.1, 1],
        "reg_lambda":        [1, 1.5, 2, 5],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=40,
        scoring="f1",
        cv=cv,
        verbose=1,
        random_state=42,
        n_jobs=-1,
    )

    print("[…] Running RandomizedSearchCV (40 iterations × 5-fold CV) …")
    search.fit(X_train, y_train)

    print(f"[✓] Best CV F1 : {search.best_score_:.4f}")
    print(f"[✓] Best params: {search.best_params_}")
    return search.best_estimator_


# =============================================================================
# 5. EVALUATION
# =============================================================================

def evaluate(model, X_test, y_test):
    """Print classification metrics and return predictions."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print("\n" + "="*50)
    print("  MODEL EVALUATION ON HELD-OUT TEST SET")
    print("="*50)
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print("="*50)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    return y_pred, y_proba, f1, auc


# =============================================================================
# 6. VISUALISATIONS
# =============================================================================

def plot_churn_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Churn Distribution Overview", fontsize=14, fontweight="bold")

    # Count plot
    churn_counts = df["Churn"].map({"Yes": "Churn", "No": "No Churn"}).value_counts()
    axes[0].bar(churn_counts.index, churn_counts.values,
                color=[COLORS["churn"], COLORS["no_churn"]])
    axes[0].set_title("Class Distribution")
    axes[0].set_ylabel("Count")
    for bar, val in zip(axes[0].patches, churn_counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                     f"{val:,}", ha="center", fontsize=10)

    # Pie chart
    axes[1].pie(churn_counts.values, labels=churn_counts.index,
                colors=[COLORS["churn"], COLORS["no_churn"]],
                autopct="%1.1f%%", startangle=140, textprops={"fontsize": 11})
    axes[1].set_title("Churn Ratio")

    plt.tight_layout()
    plt.savefig("churn_distribution.png", dpi=150, bbox_inches="tight")
    print("[✓] Saved → churn_distribution.png")
    plt.show()


def plot_feature_importance(model, feature_names: list, top_n: int = 15):
    importances = model.feature_importances_
    feat_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("Reds_r", top_n)
    sns.barplot(data=feat_df, x="Importance", y="Feature", palette=palette, ax=ax)
    ax.set_title(f"Top {top_n} Feature Importances (XGBoost)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("")

    # Annotate top 5 as "key churn drivers"
    for i, (_, row) in enumerate(feat_df.iterrows()):
        label = "★ Key Driver" if i < 5 else ""
        ax.text(row["Importance"] + 0.001, i, label,
                va="center", fontsize=8, color=COLORS["churn"])

    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
    print("[✓] Saved → feature_importance.png")
    plt.show()

    print("\n[TOP 5 CHURN DRIVERS]")
    for rank, (_, row) in enumerate(feat_df.head(5).iterrows(), 1):
        print(f"  {rank}. {row['Feature']:<25}  importance = {row['Importance']:.4f}")


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"],
                linewidths=0.5, ax=ax)
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    print("[✓] Saved → confusion_matrix.png")
    plt.show()


def plot_roc_curve(y_test, y_proba, auc: float):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color=COLORS["churn"], lw=2,
            label=f"XGBoost (AUC = {auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.08, color=COLORS["churn"])
    ax.set_title("ROC Curve — Churn Prediction", fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=150, bbox_inches="tight")
    print("[✓] Saved → roc_curve.png")
    plt.show()


def plot_churn_by_contract(df: pd.DataFrame):
    """Business insight: churn rate by contract type."""
    summary = (
        df.groupby("Contract")["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reset_index()
        .rename(columns={"Churn": "Churn Rate (%)"})
        .sort_values("Churn Rate (%)", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=summary, x="Contract", y="Churn Rate (%)",
                palette=["#e74c3c", "#e67e22", "#2ecc71"], ax=ax)
    ax.set_title("Churn Rate by Contract Type", fontsize=13, fontweight="bold")
    ax.set_ylabel("Churn Rate (%)")
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%",
                ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig("churn_by_contract.png", dpi=150, bbox_inches="tight")
    print("[✓] Saved → churn_by_contract.png")
    plt.show()


def plot_tenure_vs_churn(df: pd.DataFrame):
    """Business insight: tenure distribution by churn status."""
    fig, ax = plt.subplots(figsize=(9, 4))
    for label, color in [("Yes", COLORS["churn"]), ("No", COLORS["no_churn"])]:
        subset = df[df["Churn"] == label]["tenure"]
        ax.hist(subset, bins=30, alpha=0.6, color=color,
                label=f"{'Churned' if label=='Yes' else 'Retained'}", edgecolor="white")
    ax.set_title("Tenure Distribution by Churn Status", fontsize=13, fontweight="bold")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig("tenure_vs_churn.png", dpi=150, bbox_inches="tight")
    print("[✓] Saved → tenure_vs_churn.png")
    plt.show()


# =============================================================================
# 7. MAIN PIPELINE
# =============================================================================

def main():
    print("\n" + "="*60)
    print("   CUSTOMER CHURN CLASSIFICATION PIPELINE")
    print("="*60 + "\n")

    # ── Step 1: Load ──────────────────────────────────────────────
    df_raw = load_data()

    # ── Step 2: EDA Plots (business reporting) ────────────────────
    print("\n[Step 2] Generating EDA visualisations …")
    plot_churn_distribution(df_raw)
    if "Contract" in df_raw.columns:
        plot_churn_by_contract(df_raw)
    plot_tenure_vs_churn(df_raw)

    # ── Step 3: Preprocess ────────────────────────────────────────
    print("\n[Step 3] Preprocessing …")
    X_train, X_test, y_train, y_test, feature_names = preprocess(df_raw)

    # ── Step 4: SMOTE ─────────────────────────────────────────────
    print("\n[Step 4] Applying SMOTE …")
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    # ── Step 5: Tune + Train ──────────────────────────────────────
    print("\n[Step 5] Hyperparameter tuning …")
    best_model = tune_model(X_train_res, y_train_res)

    # ── Step 6: Evaluate ──────────────────────────────────────────
    print("\n[Step 6] Evaluating on test set …")
    y_pred, y_proba, f1, auc = evaluate(best_model, X_test, y_test)

    # ── Step 7: Model Visualisations ─────────────────────────────
    print("\n[Step 7] Generating model visualisations …")
    plot_feature_importance(best_model, feature_names, top_n=15)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_proba, auc)

    # ── Done ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"  ✓  Pipeline complete  |  F1={f1:.3f}  ROC-AUC={auc:.3f}")
    print("="*60)
    print("\nSaved artefacts:")
    artefacts = [
        "churn_distribution.png",
        "churn_by_contract.png",
        "tenure_vs_churn.png",
        "feature_importance.png",
        "confusion_matrix.png",
        "roc_curve.png",
    ]
    for a in artefacts:
        print(f"   → {a}")


if __name__ == "__main__":
    main()
