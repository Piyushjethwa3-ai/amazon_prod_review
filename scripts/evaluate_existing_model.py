import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

def main():
    # Ensure reports folder exists
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)

    # Paths (hard-coded defaults, adjust if needed)
    data_csv = os.environ.get("EVAL_DATA_CSV", "data/amazon_balanced_test.csv")
    text_col = os.environ.get("EVAL_TEXT_COL", "text")
    label_col = os.environ.get("EVAL_LABEL_COL", "sentiment")
    model_path = os.environ.get("EVAL_MODEL_PATH", "models/logistic_model.pkl")
    vectorizer_path = os.environ.get("EVAL_VECTORIZER_PATH", "models/tfidf_vectorizer.pkl")

    # Load data
    df = pd.read_csv(data_csv)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Missing required columns in dataset. Found: {list(df.columns)}")

    X_test = df[text_col].astype(str)
    y_test = df[label_col]

    # Load model and vectorizer
    model = joblib.load(model_path)

    try:
        # If model is a pipeline
        y_pred = model.predict(X_test)
    except Exception:
        # If vectorizer is separate
        vectorizer = joblib.load(vectorizer_path)
        X_vec = vectorizer.transform(X_test)
        y_pred = model.predict(X_vec)

    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Save metrics.json
    metrics_path = os.path.join(reports_dir, "metrics_existing.json")
    with open(metrics_path, "w") as f:
        json.dump({"accuracy": acc, "report": report}, f, indent=4)
    print(f"[INFO] Saved metrics to {metrics_path}")

    # Save predictions.csv
    preds_path = os.path.join(reports_dir, "predictions.csv")
    pd.DataFrame({"text": X_test, "true": y_test, "pred": y_pred}).to_csv(preds_path, index=False)
    print(f"[INFO] Saved predictions to {preds_path}")

    # Save confusion_matrix.png
    cm = confusion_matrix(y_test, y_pred, labels=sorted(df[label_col].unique()))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=sorted(df[label_col].unique()),
                yticklabels=sorted(df[label_col].unique()))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    cm_path = os.path.join(reports_dir, "confusion_matrix_existing.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"[INFO] Saved confusion matrix to {cm_path}")

if __name__ == "__main__":
    main()
