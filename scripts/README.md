# Scripts

Helper scripts for prediction and evaluation using existing pickled artifacts.

## predict_existing_model.py
Run predictions with a saved model (and vectorizer if the model is not a Pipeline).

Single text:
```bash
python scripts/predict_existing_model.py \
  --model-path models/logistic_model.pkl \
  --vectorizer-path models/tfidf_vectorizer.pkl \
  --text "Great product!"

Batch CSV:

python scripts/predict_existing_model.py \
  --model-path models/logistic_model.pkl \
  --vectorizer-path models/tfidf_vectorizer.pkl \
  --input-csv data/amazon_balanced_test.csv \
  --text-col review_text \
  --output-csv reports/predictions.csv

evaluate_existing_model.py

Evaluate accuracy/F1 and export a confusion matrix.

python scripts/evaluate_existing_model.py \
  --model-path models/logistic_model.pkl \
  --vectorizer-path models/tfidf_vectorizer.pkl \
  --data-csv data/amazon_balanced_test.csv \
  --text-col review_text \
  --label-col sentiment \
  --metrics-out reports/metrics_existing.json \
  --cm-out reports/confusion_matrix_existing.png
