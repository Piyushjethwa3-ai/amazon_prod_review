#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

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
        return hasattr(obj, "predict") and hasattr(obj, "steps")

def main():
    ap = argparse.ArgumentParser(description="Evaluate an existing model on a labeled CSV.")
    ap.add_argument("--model-path", required=True, help="Path to model pickle file")
    ap.add_argument("--vectorizer-path", help="Needed if model is not a Pipeline")
    ap.add_argument("--data-csv", required=True, help="CSV containing text + label columns")
    ap.add_argument("--text-col", required=True, help="Column name for input text")
    ap.add_argument("--label-col", required=True, help="Column name for labels")
    ap.add_argument("--metrics-out", default="reports/metrics_existing.json")
    ap.add_argument("--cm-out", default="reports/confusion_matrix_existing.png")
    args = ap.parse_args()

    model = load_pickle(args.model_path)
    vectorizer = load_pickle(args.vectorizer_path) if (args.vectorizer_path and not is_pipeline(model)) else None

    df = pd.read_csv(args.data_csv)
    if args.text_col not in df.columns or args.label_col not in df.columns:
        raise ValueError(f"Missing columns. Have: {list(df.columns)}")

    texts = df[args.text_col].astype(str)
    y_true = df[args.label_col].astype(str)

    if is_pipeline(model):
        y_pred = model.predict(texts)
    else:
        if vectorizer is None:
            raise ValueError("Model is not a Pipeline; please pass --vectorizer-path.")
        X = vectorizer.transform(texts)
        y_pred = model.predict(X)

    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    acc = float(accuracy_score(y_true, y_pred))
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump({"accuracy": acc, "report": report}, f, indent=2)

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
    plt.savefig(args.cm_out, dpi=160)

    print(f"Accuracy: {acc:.4f}")
    print(f"Saved metrics to {args.metrics_out}")
    print(f"Saved confusion matrix to {args.cm_out}")

if __name__ == "__main__":
    main()
