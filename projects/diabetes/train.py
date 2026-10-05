"""Reproduce the experiment with deduplication and separate validation/test sets."""
import json

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from preprocessing import PROJECT_DIR, encode_features


def main():
    raw = pd.read_csv(PROJECT_DIR / "data" / "diabetes_prediction_dataset.csv")
    data = raw.drop_duplicates().reset_index(drop=True)
    train, remaining = train_test_split(data, test_size=0.4, random_state=42, stratify=data.diabetes)
    validation, test = train_test_split(remaining, test_size=0.5, random_state=42, stratify=remaining.diabetes)
    model = CatBoostClassifier(
        iterations=100, learning_rate=0.1, depth=6, random_seed=42,
        verbose=False, allow_writing_files=False, thread_count=2,
    )
    model.fit(
        encode_features(train), train.diabetes,
        eval_set=(encode_features(validation), validation.diabetes),
        use_best_model=True,
    )
    prediction = model.predict(encode_features(test))
    probability = model.predict_proba(encode_features(test))[:, 1]
    metrics = {
        "raw_rows": len(raw), "duplicate_rows_removed": len(raw) - len(data),
        "train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test),
        "split_seed": 42, "threshold": 0.5, "tree_count": model.tree_count_,
        "accuracy": accuracy_score(test.diabetes, prediction),
        "precision": precision_score(test.diabetes, prediction, zero_division=0),
        "recall": recall_score(test.diabetes, prediction, zero_division=0),
        "f1": f1_score(test.diabetes, prediction, zero_division=0),
        "roc_auc": roc_auc_score(test.diabetes, probability),
        "confusion_matrix": confusion_matrix(test.diabetes, prediction).tolist(),
        "note": "One seeded holdout split on the supplied dataset; not clinical validation.",
    }
    (PROJECT_DIR / "models").mkdir(exist_ok=True)
    model.save_model(str(PROJECT_DIR / "models" / "diabetes_prediction_model.cbm"))
    reports = PROJECT_DIR / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
