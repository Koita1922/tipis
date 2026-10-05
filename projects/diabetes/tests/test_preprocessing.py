import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing import FEATURES, encode_features


def rows():
    return pd.DataFrame({
        'gender': ['Female', 'Male', 'Other', 'Female', 'Male', 'Female'],
        'age': [30]*6, 'hypertension': [0]*6, 'heart_disease': [0]*6,
        'smoking_history': ['No Info', 'current', 'ever', 'former', 'never', 'not current'],
        'bmi': [22.5]*6, 'HbA1c_level': [5.5]*6, 'blood_glucose_level': [100]*6,
    })


def test_encoding_matches_original_notebook():
    raw = rows()
    legacy = raw.copy()
    legacy.gender = legacy.gender.map({'Male': 1, 'Female': 0})
    legacy.smoking_history = legacy.smoking_history.astype('category').cat.codes
    pd.testing.assert_frame_equal(encode_features(raw), legacy, check_dtype=False)


def test_inference_order_and_no_input_mutation():
    raw = rows()[list(reversed(FEATURES))]
    before = raw.copy(deep=True)
    assert list(encode_features(raw).columns) == FEATURES
    pd.testing.assert_frame_equal(raw, before)


def test_unknown_category_rejected():
    raw = rows()
    raw.loc[0, 'smoking_history'] = 'unexpected'
    with pytest.raises(ValueError):
        encode_features(raw)
