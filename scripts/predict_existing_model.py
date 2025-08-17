#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys, json
import pandas as pd

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

def run_single(model, text: str, vectorizer=None):
    if is_pipeline(model):
        return model.predict([text])[0]
    if vectorizer is None:
        raise ValueError("Model not a Pipeline; pass --vectorizer-path.")
    X = vectorizer.transform([text])
    return model.predict(X)[0]

def run_batch(model, df: pd.DataFrame, text_col: str, vectorizer=None):
    if text_col not in df.columns:
        raise ValueError(f"'{text_col}' not found in columns: {list(df.columns)}")
    texts = df[text_col].astype(str)
    if is_pipeline(model):
        preds = model.predict(texts)
    else:
        if vectorizer is None:
            raise ValueError("Model not a Pipeline; pass --vectorizer-path.")
        X = vectorizer.transform(texts)
        preds = model.predict(X)
    return pd.Series(preds, index=df.index)

def main():
    ap = argparse.ArgumentParser(description="Predict using an existing pickled model.")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--vectorizer-path", help="Needed only if model is NOT a Pipeline")
    ap.add_argument("--text", help="Single text to classify")
    ap.add_argument("--input-csv", help="CSV path for batch predictions")
    ap.add_argument("--text-col", help="Text column name for batch predictions")
    ap.add_argument("--output-csv", default="reports/predictions.csv")
    ap.add_argument("--print-proba", action="store_true")
    ap.add_argument("--labels-out", default="")
    args = ap.parse_args()

    model = load_pickle(args.model_path)
    vectorizer = load_pickle(args.vectorizer_path) if (args.vectorizer_path and not is_pipeline(model)) else None

    if args.labels_out:
        labels = {}
        if hasattr(model, "classes_"):
            labels["classes_"] = list(model.classes_)
        elif is_pipeline(model) and hasattr(model[-1], "classes_"):
            labels["classes_"] = list(model[-1].classes_)
        with open(args.labels_out, "w") as f:
            json.dump(labels, f, indent=2)

    if args.text:
        pred = run_single(model, args.text, vectorizer)
        print(pred)
        if args.print_proba and hasattr(model, "predict_proba"):
            if is_pipeline(model):
                proba = model.predict_proba([args.text])[0]
            else:
                X = vectorizer.transform([args.text])
                proba = model.predict_proba(X)[0]
            try:
                print("proba:", proba.tolist())
            except Exception:
                print("proba:", proba)
        return

    if args.input_csv and args.text_col:
        df = pd.read_csv(args.input_csv)
        preds = run_batch(model, df, args.text_col, vectorizer)
        out = pd.DataFrame({"id": df.index, args.text_col: df[args.text_col].astype(str), "prediction": preds})
        out.to_csv(args.output_csv, index=False)
        print(f"Saved predictions to {args.output_csv}")
        return

    ap.print_help()
    sys.exit(1)

if __name__ == "__main__":
    main()
