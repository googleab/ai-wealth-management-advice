# tests/test_pipeline.py
import pytest
import pandas as pd
from src.data_quality import load_data, validate_data
from src.data_processing import load_and_preprocess_data

def test_data_quality():
    df = load_data('data/client_data.csv')
    results = validate_data(df)
    assert results["success"] is True, "Data quality validation failed."

def test_preprocessing_shape():
    X_processed, y, pipeline = load_and_preprocess_data('data/client_data.csv')
    # Ensure that output has the expected number of rows (should equal number of original rows)
    df = pd.read_csv('data/client_data.csv')
    assert X_processed.shape[0] == df.shape[0], "Mismatch in number of rows after preprocessing."

if __name__ == '__main__':
    pytest.main()
