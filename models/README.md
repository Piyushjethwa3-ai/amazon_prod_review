# Models

Serialized model artifacts for inference:

- `logistic_model.pkl` – trained classifier
- `tfidf_vectorizer.pkl` – fitted TF-IDF vectorizer (used if the model is not a Pipeline)

**Tips**
- Prefer saving a scikit-learn Pipeline (vectorizer + model) to avoid separate vectorizer handling.
- In production, store models in an artifact store (S3, GCS) rather than Git.

