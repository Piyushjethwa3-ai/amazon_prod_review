#!/usr/bin/env python3
from __future__ import annotations
import os, json
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Defaults (override via env if you ever need to)
DATA_CSV = os.getenv("EVAL_DATA_CSV", "data/amazon_balanced_test.csv")
TEXT_COL = os.getenv("EVAL_TEXT_COL", "text")
LABEL_COL = os.getenv("EVAL_LABEL_COL", "sentiment")
MODEL_PATH = os.getenv("EVAL_MODEL_PATH", "models/logistic_model.pkl")
VECTORIZER_PATH = os.getenv("EVAL_VECTORIZER_PATH", "models/tfidf_vectorizer.pkl")

REPORTS = Path("reports")
METRICS_OUT = REPORTS / "metrics_existing.json"
CM_OUT = REPORTS / "confusion_matrix_existing.png"
PRED_OUT = REPORTS / "predictions.csv"

def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    # --- Robust CSV read ---
    print(f"[INFO] Reading dataset from {DATA_CSV}")
    try:
        df = pd.read_csv(DATA_CSV, quotechar='"')
    except Exception as e:
        print(f"[WARN] Default read_csv failed: {e}")
        print("[WARN] Retrying with on_bad_lines='skip'")
        df = pd.read_csv(DATA_CSV, quotechar='"', on_bad_lines="skip")

    # Log dataset preview in CI logs
    print("[INFO] First 3 rows of dataset:")
    print(df.head(3).to_string())
    print("[INFO] Columns detected:", list(df.columns))

    if TEXT_COL not in df.columns or LABEL_COL not in df.columns:
        raise ValueError(f"Missing required columns: need '{TEXT_COL}' and '{LABEL_COL}'. Found: {list(df.columns)}")

    X = df[TEXT_COL].astype(str)
    y = df[LABEL_COL].astype(str)

    # Load model (pipeline or estimator)
    model = joblib.load(MODEL_PATH)
    try:
        y_pred = model.predict(X)  # pipeline case
    except Exception:
        vec = joblib.load(VECTORIZER_PATH)  # estimator+vectorizer case
        Xv = vec.transform(X)
        y_pred = model.predict(Xv)

    # Metrics
    acc = float(accuracy_score(y, y_pred))
    report = classification_report(y, y_pred, output_dict=True)
    with open(METRICS_OUT, "w") as f:
        json.dump({"accuracy": acc, "report": report}, f, indent=2)

    # Predictions CSV
    pd.DataFrame({TEXT_COL: X, "true": y, "pred": y_pred}).to_csv(PRED_OUT, index=False)

    # Confusion matrix (pure matplotlib, no seaborn)
    labels = sorted(df[LABEL_COL].astype(str).unique())
    cm = confusion_matrix(y, y_pred, labels=labels)
    fig, ax = plt.subplots()
    im = ax.imshow(cm)  # default colormap; no explicit colors
    ax.set_title("Confusion Matrix (Existing Model)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(CM_OUT, dpi=160)

    print(f"SAVED: {METRICS_OUT}")
    print(f"SAVED: {PRED_OUT}")
    print(f"SAVED: {CM_OUT}")
    print(f"ACCURACY: {acc:.4f}")

if __name__ == "__main__":
    main()
