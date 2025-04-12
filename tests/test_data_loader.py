import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os
import tempfile

# Assume data_loader.py exists with the following functions:
# We define them here for the test file to be self-contained and runnable.

# --- Start: Hypothetical data_loader module content ---
def load_csv(file_path):
    """Loads data from a CSV file."""
    if not isinstance(file_path, str) or not file_path.lower().endswith('.csv'):
        raise ValueError("File path must be a string ending with .csv")
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            # Check if the file itself was empty or just had headers
            with open(file_path, 'r') as f:
                content = f.read().strip()
                if not content or all(line.strip() == '' or i == 0 for i, line in enumerate(content.splitlines())):
                     raise ValueError("CSV file is empty or contains only headers")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except pd.errors.EmptyDataError:
         raise ValueError("CSV file is empty or contains only headers")
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV: {e}")


def handle_missing_values(df, strategy='mean', subset=None):
    """Handles missing values in a DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    df_copy = df.copy()

    if subset:
        if not isinstance(subset, list) or not all(isinstance(col, str) for col in subset):
             raise TypeError("Subset must be a list of column names (strings)")
        if not all(col in df_copy.columns for col in subset):
            raise ValueError("Subset contains columns not present in the DataFrame")
        columns_to_process = subset
    else:
        columns_to_process = df_copy.columns

    numeric_cols_in_subset = df_copy[columns_to_process].select_dtypes(include=np.number).columns
    
    if strategy == 'mean':
        for col in numeric_cols_in_subset:
            if df_copy[col].isnull().any():
                mean_val = df_copy[col].mean()
                df_copy[col].fillna(mean_val, inplace=True)
    elif strategy == 'median':
         for col in numeric_cols_in_subset:
            if df_copy[col].isnull().any():
                median_val = df_copy[col].median()
                df_copy[col].fillna(median_val, inplace=True)
    elif strategy == 'mode':
        # Applicable to both numeric and non-numeric
        cols_for_mode = [col for col in columns_to_process if df_copy[col].isnull().any()]
        for col in cols_for_mode:
             mode_val = df_copy[col].mode()
             if not mode_val.empty: # Handle cases where mode is ambiguous or column is all NaN
                 df_copy[col].fillna(mode_val[0], inplace=True)
    elif strategy == 'drop':
        df_copy.dropna(subset=columns_to_process, inplace=True)
    elif strategy == 'constant':
        raise ValueError("Strategy 'constant' requires a 'fill_value' argument")
    elif callable(strategy):
         for col in columns_to_process:
             if df_copy[col].isnull().any():
                 df_copy[col] = strategy(df_copy[col])
    else:
        raise ValueError("Invalid strategy. Choose 'mean', 'median', 'mode', 'drop', or provide a callable.")

    return df_copy

def scale_numerical_features(df, columns=None):
    """Scales numerical features using StandardScaler."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    df_copy = df.copy()

    if columns is None:
        numerical_columns = df_copy.select_dtypes(include=np.number).columns.tolist()
    else:
        if not isinstance(columns, list) or not all(isinstance(col, str) for col in columns):
             raise TypeError("Columns must be a list of column names (strings)")
        if not all(col in df_copy.columns for col in columns):
            raise ValueError("Columns list contains names not present in the DataFrame")
        if not all(pd.api.types.is_numeric_dtype(df_copy[col]) for col in columns):
             raise ValueError("Specified columns for scaling must be numeric")
        numerical_columns = columns

    if not numerical_columns:
        # Return original df and None if no numerical columns found/specified
        return df_copy, None

    scaler = StandardScaler()
    # Ensure we only scale columns that actually exist and are numeric
    valid_numeric_columns = [col for col in numerical_columns if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col])]

    if not valid_numeric_columns:
         return df_copy, None

    # Handle potential NaNs before scaling if necessary, or raise error
    if df_copy[valid_numeric_columns].isnull().values.any():
        raise ValueError("Numerical columns to be scaled contain NaN values. Handle missing values first.")

    df_copy[valid_numeric_columns] = scaler.fit_transform(df_copy[valid_numeric_columns])
    return df_copy, scaler

def encode_categorical_features(df, columns=None, drop_original=True):
    """Encodes categorical features using OneHotEncoder."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    df_copy = df.copy()

    if columns is None:
        categorical_columns = df_copy.select_dtypes(include=['object', 'category']).columns.tolist()
    else:
        if not isinstance(columns, list) or not all(isinstance(col, str) for col in columns):
             raise TypeError("Columns must be a list of column names (strings)")
        if not all(col in df_copy.columns for col in columns):
            raise ValueError("Columns list contains names not present in the DataFrame")
        # Check if specified columns are actually categorical or object type
        are_categorical = all(pd.api.types.is_object_dtype(df_copy[col]) or pd.api.types.is_categorical_dtype(df_copy[col]) for col in columns)
        if not are_categorical:
             raise ValueError("Specified columns for encoding must be of object or category dtype")
        categorical_columns = columns

    if not categorical_columns:
        # Return original df and None if no categorical columns found/specified
        return df_copy, None

    # Handle potential NaNs before encoding if necessary
    # OneHotEncoder can handle NaNs if configured, but default is to error.
    # Let's explicitly handle them or ensure they are handled beforehand.
    # For simplicity here, we assume NaNs might cause issues or are handled prior.
    # Alternatively, fill NaNs in categorical columns with a placeholder like 'missing'
    for col in categorical_columns:
        if df_copy[col].isnull().any():
            # Option 1: Raise error
            # raise ValueError(f"Categorical column '{col}' contains NaN values. Handle missing values first.")
            # Option 2: Fill with a placeholder (more robust for OHE)
            df_copy[col].fillna('Missing', inplace=True)


    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    # Ensure we only encode columns that exist and are categorical/object
    valid_categorical_columns = [col for col in categorical_columns if col in df_copy.columns and (pd.api.types.is_object_dtype(df_copy[col]) or pd.api.types.is_categorical_dtype(df_copy[col]))]

    if not valid_categorical_columns:
        return df_copy, None

    encoded_data = encoder.fit_transform(df_copy[valid_categorical_columns])

    # Create new DataFrame with encoded columns
    encoded_df = pd.DataFrame(encoded_data, index=df_copy.index, columns=encoder.get_feature_names_out(valid_categorical_columns))

    if drop_original:
        df_copy = df_copy.drop(columns=valid_categorical_columns)

    df_copy = pd.concat([df_copy, encoded_df], axis=1)

    return df_copy, encoder

# --- End: Hypothetical data_loader module content ---


# --- Start: Test Fixtures ---

@pytest.fixture
def sample_dataframe():
    """Provides a sample pandas DataFrame for testing."""
    data = {
        'numeric_col_1': [1.0, 2.0, np.nan, 4.0, 5.0],
        'numeric_col_2': [10, 20, 30, 40, 50],
        'categorical_col_1': ['A', 'B', 'A', 'C', 'B'],
        'categorical_col_2': ['X', 'Y', np.nan, 'X', 'Y'],
        'mixed_col': [1, 'text', 3.0, np.nan, 'more_text']
    }
    return pd.DataFrame(data)

@pytest.fixture
def temp_csv_file(tmp_path):
    """Creates a temporary CSV file for testing loading functions."""
    file_path = tmp_path / "test_data.csv"
    data = {'col1': [1, 2, 3], 'col2': ['A', 'B', 'C']}
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    return str(file_path) # Return path as string

@pytest.fixture
def temp_empty_csv_file(tmp_path):
    """Creates an empty temporary CSV file."""
    file_path = tmp_path / "empty_data.csv"
    file_path.touch()
    return str(file_path)

@pytest.fixture
def temp_header_only_csv_file(tmp_path):
    """Creates a temporary CSV file with only headers."""
    file_path = tmp_path / "header_only_data.csv"
    with open(file_path, 'w') as f:
        f.write("header1,header2\n")
    return str(file_path)

# --- End: Test Fixtures ---


# --- Start: Test Cases ---

# 1. Data Loading Tests
def test_load_csv_success(temp_csv_file):
    """Tests successful loading of a valid CSV file."""
    df = load_csv(temp_csv_file)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns) == ['col1', 'col2']
    assert len(df) == 3

def test_load_csv_file_not_found():
    """Tests loading a non-existent CSV file."""
    with pytest.raises(FileNotFoundError):
        load_csv("non_existent_file.csv")

def test_load_csv_empty_file(temp_empty_csv_file):
    """Tests loading an empty CSV file."""
    with pytest.raises(ValueError, match="CSV file is empty or contains only headers"):
        load_csv(temp_empty_csv_file)

def test_load_csv_header_only_file(temp_header_only_csv_file):
    """Tests loading a CSV file with only headers."""
    with pytest.raises(ValueError, match="CSV file is empty or contains only headers"):
        load_csv(temp_header_only_csv_file)


def test_load_csv_invalid_path_type():
    """Tests loading with a non-string path."""
    with pytest.raises(ValueError, match="File path must be a string ending with .csv"):
        load_csv(123)

def test_load_csv_wrong_extension():
    """Tests loading a file without a .csv extension."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmpfile:
        tmpfile.write(b"col1,col2\n1,A")
        file_path = tmpfile.name
    
    with pytest.raises(ValueError, match="File path must be a string ending with .csv"):
        load_csv(file_path)
    
    os.remove(file_path) # Clean up the temporary file


# 2. Data Preprocessing Tests (Missing Values)
def test_handle_missing_values_mean(sample_dataframe):
    """Tests handling missing values using mean strategy."""
    df_processed = handle_missing_values(sample_dataframe, strategy='mean', subset=['numeric_col_1'])
    assert not df_processed['numeric_col_1'].isnull().any()
    # Mean of [1, 2, 4, 5] is 3.0
    assert df_processed.loc[2, 'numeric_col_1'] == pytest.approx(3.0)
    # Check other columns are untouched if not in subset
    assert df_processed['numeric_col_2'].equals(sample_dataframe['numeric_col_2'])
    assert df_processed['categorical_col_2'].isnull().sum() == 1 # NaN should still be there

def test_handle_missing_values_median(sample_dataframe):
    """Tests handling missing values using median strategy."""
    df_processed = handle_missing_values(sample_dataframe, strategy='median', subset=['numeric_col_1'])
    assert not df_processed['numeric_col_1'].isnull().any()
    # Median of [1, 2, 4, 5] is 3.0
    assert df_processed.loc[2, 'numeric_col_1'] == pytest.approx(3.0)

def test_handle_missing_values_mode(sample_dataframe):
    """Tests handling missing values using mode strategy."""
    df_processed = handle_missing_values(sample_dataframe.copy(), strategy='mode', subset=['categorical_col_1', 'categorical_col_2'])
    # Mode of ['A', 'B', 'A', 'C', 'B'] is 'A' or 'B' (pandas returns first: 'A') - no NaNs here
    # Mode of ['X', 'Y', nan, 'X', 'Y'] is 'X' or 'Y' (pandas returns first: 'X')
    assert not df_processed['categorical_col_2'].isnull().any()
    assert df_processed.loc[2, 'categorical_col_2'] == 'X' # Mode of ['X', 'Y', 'X', 'Y'] is 'X' or 'Y', pandas mode() returns both, fillna uses first
    assert df_processed['numeric_col_1'].isnull().sum() == 1 # Numeric NaN should be untouched

def test_handle_missing_values_drop(sample_dataframe):
    """Tests handling missing values using drop strategy."""
    df_processed = handle_missing_values(sample_dataframe, strategy='drop', subset=['numeric_col_1', 'categorical_col_2'])
    # Rows with NaN in numeric_col_1 (index 2) or categorical_col_2 (index 2) should be dropped
    assert len(df_processed) == 3
    assert 2 not in df_processed.index
    assert not df_processed['numeric_col_1'].isnull().any()
    assert not df_processed['categorical_col_2'].isnull().any()

def test_handle_missing_values_all_columns(sample_dataframe):
    """Tests handling missing values across all applicable columns without subset."""
    df_processed = handle_missing_values(sample_dataframe.copy(), strategy='mean') # Default applies to numeric
    assert not df_processed['numeric_col_1'].isnull().any()
    assert df_processed['numeric_col_2'].equals(sample_dataframe['numeric_col_2']) # No NaNs here
    assert df_processed['categorical_col_2'].isnull().sum() == 1 # Categorical NaN untouched by 'mean'
    assert df_processed['mixed_col'].equals(sample_dataframe['mixed_col']) # Mixed type untouched by 'mean'

def test_handle_missing_values_invalid_strategy(sample_dataframe):
    """Tests using an invalid strategy."""
    with pytest.raises(ValueError, match="Invalid strategy"):
        handle_missing_values(sample_dataframe, strategy='invalid_strategy')

def test_handle_missing_values_invalid_input_type():
    """Tests passing non-DataFrame input."""
    with pytest.raises(TypeError, match="Input must be a pandas DataFrame"):
        handle_missing_values([1, 2, 3])

def test_handle_missing_values_invalid_subset_type(sample_dataframe):
    """Tests passing invalid subset type."""
    with pytest.raises(TypeError, match="Subset must be a list of column names"):
        handle_missing_values(sample_dataframe, subset="numeric_col_1")

def test_handle_missing_values_invalid_subset_column(sample_dataframe):
    """Tests passing non-existent column in subset."""
    with pytest.raises(ValueError, match="Subset contains columns not present"):
        handle_missing_values(sample_dataframe, subset=['numeric_col_1', 'non_existent_col'])


# 3. Data Transformation Tests (Scaling)
def test_scale_numerical_features_all(sample_dataframe):
    """Tests scaling all numerical features."""
    df_no_nan = handle_missing_values(sample_dataframe.copy(), strategy='mean') # Ensure no NaNs
    df_scaled, scaler = scale_numerical_features(df_no_nan)

    assert isinstance(scaler, StandardScaler)
    assert 'numeric_col_1' in df_scaled.columns
    assert 'numeric_col_2' in df_scaled.columns
    # Check if means are close to 0 and std devs close to 1
    assert np.allclose(df_scaled[['numeric_col_1', 'numeric_col_2']].mean(), 0.0, atol=1e-7)
    assert np.allclose(df_scaled[['numeric_col_1', 'numeric_col_2']].std(ddof=0), 1.0, atol=1e-7)
    # Check non-numeric columns are untouched
    assert df_scaled['categorical_col_1'].equals(df_no_nan['categorical_col_1'])

def test_scale_numerical_features_subset(sample_dataframe):
    """Tests scaling a subset of numerical features."""
    df_no_nan = handle_missing_values(sample_dataframe.copy(), strategy='mean') # Ensure no NaNs
    df_scaled, scaler = scale_numerical_features(df_no_nan, columns=['numeric_col_1'])

    assert isinstance(scaler, StandardScaler)
    assert 'numeric_col_1' in df_scaled.columns
    assert 'numeric_col_2' in df_scaled.columns
    # Check if mean of scaled column is close to 0 and std dev close to 1
    assert np.allclose(df_scaled['numeric_col_1'].mean(), 0.0, atol=1e-7)
    assert np.allclose(df_scaled['numeric_col_1'].std(ddof=0), 1.0, atol=1e-7)
    # Check other numeric column is untouched
    assert df_scaled['numeric_col_2'].equals(df_no_nan['numeric_col_2'])

def test_scale_numerical_features_no_numeric(sample_dataframe):
    """Tests scaling when there are no numerical features."""
    df_categorical = sample_dataframe[['categorical_col_1', 'categorical_col_2']].copy()
    df_scaled, scaler = scale_numerical_features(df_categorical)
    assert scaler is None
    assert df_scaled.equals(df_categorical) # DataFrame should be unchanged

def test_scale_numerical_features_with_nan(sample_dataframe):
    """Tests scaling attempt on columns with NaN values."""
    with pytest.raises(ValueError, match="Numerical columns to be scaled contain NaN values"):
        scale_numerical_features(sample_dataframe, columns=['numeric_col_1'])

def test_scale_numerical_features_invalid_input_type():
    """Tests passing non-DataFrame input to scaling."""
    with pytest.raises(TypeError, match="Input must be a pandas DataFrame"):
        scale_numerical_features([1, 2, 3])

def test_scale_numerical_features_invalid_columns_type(sample_dataframe):
    """Tests passing invalid columns type to scaling."""
    with pytest.raises(TypeError, match="Columns must be a list of column names"):
        scale_numerical_features(sample_dataframe, columns="numeric_col_1")

def test_scale_numerical_features_invalid_columns_name(sample_dataframe):
    """Tests passing non-existent column name to scaling."""
    with pytest.raises(ValueError, match="Columns list contains names not present"):
        scale_numerical_features(sample_dataframe, columns=['numeric_col_1', 'non_existent'])

def test_scale_numerical_features_non_numeric_column_in_list(sample_dataframe):
    """Tests passing non-numeric column in the scaling list."""
    with pytest.raises(ValueError, match="Specified columns for scaling must be numeric"):
        scale_numerical_features(sample_dataframe, columns=['numeric_col_1', 'categorical_col_1'])


# 4. Data Transformation Tests (Encoding)
def test_encode_categorical_features_all(sample_dataframe):
    """Tests encoding all categorical/object features."""
    df_copy = sample_dataframe.copy()
    # Fill NaN in categorical col 2 before encoding
    # df_copy['categorical_col_2'].fillna('Missing', inplace=True) # Function handles this now

    df_encoded, encoder = encode_categorical_features(df_copy)

    assert isinstance(encoder, OneHotEncoder)
    # Check original categorical columns are dropped (default)
    assert 'categorical_col_1' not in df_encoded.columns
    assert 'categorical_col_2' not in df_encoded.columns
    # Check new encoded columns exist
    assert 'categorical_col_1_A' in df_encoded.columns
    assert 'categorical_col_1_B' in df_encoded.columns
    assert 'categorical_col_1_C' in df_encoded.columns
    assert 'categorical_col_2_X' in df_encoded.columns
    assert 'categorical_col_2_Y' in df_encoded.columns
    assert 'categorical_col_2_Missing' in df_encoded.columns # NaN was filled with 'Missing'
    # Check numeric columns are untouched
    assert df_encoded['numeric_col_1'].equals(df_copy['numeric_col_1'])
    # Check shape consistency
    assert df_encoded.shape[0] == df_copy.shape[0]
    # Expected number of columns: original numeric (2) + original mixed (1) + encoded cat1 (3) + encoded cat2 (3) = 9
    assert df_encoded.shape[1] == 9


def test_encode_categorical_features_subset(sample_dataframe):
    """Tests encoding a subset of categorical features."""
    df_copy = sample_dataframe.copy()
    # df_copy['categorical_col_2'].fillna('Missing', inplace=True) # Function handles this now

    df_encoded, encoder = encode_categorical_features(df_copy, columns=['categorical_col_1'])

    assert isinstance(encoder, OneHotEncoder)
    assert 'categorical_col_1' not in df_encoded.columns # Dropped
    assert 'categorical_col_2' in df_encoded.columns # Not encoded, should remain (with NaN)
    assert 'categorical_col_1_A' in df_encoded.columns
    assert 'categorical_col_1_B' in df_encoded.columns
    assert 'categorical_col_1_C' in df_encoded.columns
    assert 'categorical_col_2_X' not in df_encoded.columns # Not encoded
    assert df_encoded['categorical_col_2'].equals(df_copy['categorical_col_2']) # Check it's unchanged

def test_encode_categorical_features_keep_original(sample_dataframe):
    """Tests encoding while keeping original categorical columns."""
    df_copy = sample_dataframe.copy()
    # df_copy['categorical_col_2'].fillna('Missing', inplace=True) # Function handles this now

    df_encoded, encoder = encode_categorical_features(df_copy, columns=['categorical_col_1'], drop_original=False)

    assert isinstance(encoder, OneHotEncoder)
    assert 'categorical_col_1' in df_encoded.columns # Kept
    assert 'categorical_col_1_A' in df_encoded.columns
    assert 'categorical_col_1_B' in df_encoded.columns
    assert 'categorical_col_1_C' in df_encoded.columns

def test_encode_categorical_features_no_categorical(sample_dataframe):
    """Tests encoding when there are no categorical features."""
    df_numeric = sample_dataframe[['numeric_col_1', 'numeric_col_2']].copy()
    df_encoded, encoder = encode_categorical_features(df_numeric)
    assert encoder is None
    assert df_encoded.equals(df_numeric) # DataFrame should be unchanged

# Test encoding handles NaNs by filling with 'Missing' placeholder
def test_encode_categorical_features_handles_nan(sample_dataframe):
    """Tests that encoding handles NaN values by creating a 'Missing' category."""
    df_copy = sample_dataframe.copy()
    df_encoded, encoder = encode_categorical_features(df_copy, columns=['categorical_col_2'])

    assert isinstance(encoder, OneHotEncoder)
    assert 'categorical_col_2' not in df_encoded.columns # Original dropped
    assert 'categorical_col_2_X' in df_encoded.columns
    assert 'categorical_col_2_Y' in df_encoded.columns
    assert 'categorical_col_2_Missing' in df_encoded.columns # Check the NaN category
    # Verify the row where NaN was present (index 2)
    assert df_encoded.loc[2, 'categorical_col_2_Missing'] == 1
    assert df_encoded.loc[2, 'categorical_col_2_X'] == 0
    assert df_encoded.loc[2, 'categorical_col_2_Y'] == 0


def test_encode_categorical_features_invalid_input_type():
    """Tests passing non-DataFrame input to encoding."""
    with pytest.raises(TypeError, match="Input must be a pandas DataFrame"):
        encode_categorical_features([1, 2, 3])

def test_encode_categorical_features_invalid_columns_type(sample_dataframe):
    """Tests passing invalid columns type to encoding."""
    with pytest.raises(TypeError, match="Columns must be a list of column names"):
        encode_categorical_features(sample_dataframe, columns="categorical_col_1")

def test_encode_categorical_features_invalid_columns_name(sample_dataframe):
    """Tests passing non-existent column name to encoding."""
    with pytest.raises(ValueError, match="Columns list contains names not present"):
        encode_categorical_features(sample_dataframe, columns=['categorical_col_1', 'non_existent'])

def test_encode_categorical_features_numeric_column_in_list(sample_dataframe):
    """Tests passing numeric column in the encoding list."""
    with pytest.raises(ValueError, match="Specified columns for encoding must be of object or category dtype"):
        encode_categorical_features(sample_dataframe, columns=['categorical_col_1', 'numeric_col_1'])

# --- End: Test Cases ---