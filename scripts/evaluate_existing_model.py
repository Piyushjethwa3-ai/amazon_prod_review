#!/usr/bin/env python3
from __future__ import annotations
import os, json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Configuration (overridable via env vars) ----------
DATA_CSV       = os.getenv("EVAL_DATA_CSV", "data/amazon_balanced_test.csv")
TEXT_COL_ENV   = os.getenv("EVAL_TEXT_COL", "")        # leave blank to auto-detect
LABEL_COL_ENV  = os.getenv("EVAL_LABEL_COL", "sentiment")
MODEL_PATH     = os.getenv("EVAL_MODEL_PATH", "models/logistic_model.pkl")
VECTORIZER_PATH= os.getenv("EVAL_VECTORIZER_PATH", "models/tfidf_vectorizer.pkl")

REPORTS_DIR    = Path("reports")
METRICS_OUT    = REPORTS_DIR / "metrics_existing.json"
CM_OUT         = REPORTS_DIR / "confusion_matrix_existing.png"
PRED_OUT       = REPORTS_DIR / "predictions.csv"
# --------------------------------------------------------------

def load_pickle(path: str):
    import joblib, pickle
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

def is_pipeline(obj) -> bool:
    try:
        from sklearn.pipeline import Pipeline
        return isinstance(obj, Pipeline)
    except Exception:
        # Robust fallback (still lets us predict if API matches)
        return hasattr(obj, "predict") and hasattr(obj, "fit") and hasattr(obj, "get_params")

def autodetect_cols(df: pd.DataFrame, text_col_env: str, label_col_env: str) -> tuple[str, str]:
    # If user provided, trust those (validate)
    if label_col_env in df.columns:
        label_col = label_col_env
    else:
        # Try common label names
        candidates = ["sentiment", "label", "target", "y", "labels"]
        label_col = next((c for c in candidates if c in df.columns), None)
        if label_col is None:
            raise ValueError(f"Could not find label column. Have: {list(df.columns)}")

    if text_col_env and text_col_env in df.columns:
        text_col = text_col_env
    else:
        # Try common text names
        text_candidates = ["text", "review_text", "review", "content", "body", "comment"]
        text_col = next((c for c in text_candidates if c in df.columns and c != label_col), None)
        if text_col is None:
            # As a last resort: if there are exactly 2 columns, choose the non-label as text
            if len(df.columns) == 2 and label_col in df.columns:
                text_col = next(c for c in df.columns if c != label_col)
            else:
                raise ValueError(
                    f"Could not find text column. Tried {text_candidates}. Have: {list(df.columns)}"
                )
    return text_col, label_col

def main():
    # Ensure output dir
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_pickle(MODEL_PATH)

    # Load vectorizer only if NOT a pipeline
    vectorizer = None
    if not is_pipeline(model):
        if not Path(VECTORIZER_PATH).exists():
            raise FileNotFoundError(
                f"Model is not a Pipeline, and vectorizer not found at {VECTORIZER_PATH}."
            )
        vectorizer = load_pickle(VECTORIZER_PATH)

    # Load data
    if not Path(DATA_CSV).exists():
        raise FileNotFoundError(f"Data CSV not found at {DATA_CSV}")
    df = pd.read_csv(DATA_CSV)

    # Detect columns
    text_col, label_col = autodetect_cols(df, TEXT_COL_ENV, LABEL_COL_ENV)

    texts = df[text_col].astype(str)
    y_true = df[label_col].astype(str)

    # Predict
    if is_pipeline(model):
        y_pred = model.predict(texts)
    else:
        X = vectorizer.transform(texts)
        y_pred = model.predict(X)

    # Metrics
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    acc = float(accuracy_score(y_true, y_pred))
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    # Save metrics JSON
    with open(METRICS_OUT, "w") as f:
        json.dump({"accuracy": acc, "report": report}, f, indent=2)

    # Save confusion matrix image (no custom colors)
    import numpy as np
    fig, ax = plt.subplots()
    im = ax.imshow(cm)
    ax.set_title("Confusion Matrix (Existing Model)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(CM_OUT, dpi=160)

    # Save predictions CSV
    out_df = pd.DataFrame({"id": df.index, text_col: texts, "prediction": y_pred})
    out_df.to_csv(PRED_OUT, index=False)

    # Helpful CI logs
    print(f"SAVED: {METRICS_OUT}")
    print(f"SAVED: {CM_OUT}")
    print(f"SAVED: {PRED_OUT}")
    print(f"ACCURACY: {acc:.4f}")
    print(f"TEXT_COL: {text_col}  LABEL_COL: {label_col}")

if __name__ == "__main__":
    main()
