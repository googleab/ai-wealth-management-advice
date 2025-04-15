# src/data_quality.py
import pandas as pd
import great_expectations as ge

def load_data(filepath: str) -> pd.DataFrame:
    """Load the CSV dataset."""
    df = pd.read_csv(filepath)
    return df

def validate_data(df: pd.DataFrame) -> dict:
    """Validate the dataset using Great Expectations."""
    # Convert DataFrame to a GE dataframe
    df_ge = ge.from_pandas(df)
    
    # Define expected schema and rules
    expected_columns = ['age', 'income', 'AUM', 'num_transactions', 'engagement_score', 'Risk_Label']
    df_ge.expect_table_columns_to_match_ordered_list(expected_columns)
    
    # Expect no nulls
    for col in expected_columns:
        df_ge.expect_column_values_to_not_be_null(col)
    
    # Expect age within realistic range
    df_ge.expect_column_values_to_be_between('age', min_value=18, max_value=100)
    df_ge.expect_column_values_to_be_between('income', min_value=0)
    df_ge.expect_column_values_to_be_between('AUM', min_value=0)
    df_ge.expect_column_values_to_be_between('num_transactions', min_value=0)
    df_ge.expect_column_values_to_be_between('engagement_score', min_value=0, max_value=1)
    df_ge.expect_column_values_to_be_in_set('Risk_Label', [0, 1])
    
    return df_ge.validate()

if __name__ == '__main__':
    filepath = 'data/client_data.csv'
    df = load_data(filepath)
    results = validate_data(df)
    print("Data Quality Validation Results:")
    print(results)
