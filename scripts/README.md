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

