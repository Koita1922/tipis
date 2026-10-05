"""Shared feature order and category encoding for training and inference."""
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
FEATURES = [
    "gender", "age", "hypertension", "heart_disease", "smoking_history",
    "bmi", "HbA1c_level", "blood_glucose_level",
]
# Match pandas categorical codes in the original notebook, exactly.
SMOKING_CODES = {
    "No Info": 0, "current": 1, "ever": 2,
    "former": 3, "never": 4, "not current": 5,
}
GENDER_CODES = {"Female": 0, "Male": 1}


def encode_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Encode raw dataset categories without depending on category sort order."""
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")
    result = frame.loc[:, FEATURES].copy()
    if not result["gender"].isin(["Female", "Male", "Other"]).all():
        raise ValueError("Unknown gender category")
    if not result["smoking_history"].isin(SMOKING_CODES).all():
        raise ValueError("Unknown smoking history category")
    # Original notebook maps Other to NaN; CatBoost supports numeric NaN.
    result["gender"] = result["gender"].map(GENDER_CODES)
    result["smoking_history"] = result["smoking_history"].map(SMOKING_CODES)
    for column in FEATURES:
        result[column] = pd.to_numeric(result[column], errors="raise")
    return result
