import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, List, Optional, Union

# Constants
NUM_LANDMARKS = 21
WRIST = 0
THUMB_TIP = 4
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_TIP = 12
RING_FINGER_TIP = 16
PINKY_TIP = 20

FINGERTIP_INDICES = [
    THUMB_TIP,
    INDEX_FINGER_TIP,
    MIDDLE_FINGER_TIP,
    RING_FINGER_TIP,
    PINKY_TIP,
]

def load_data_from_csv(
    file_path: Union[str, Path]
) -> Optional[pd.DataFrame]:
    """Loads gesture data from a single CSV file."""
    try:
        path = Path(file_path)
        if not path.is_file():
            print(f"Warning: File not found at {file_path}")
            return None
        df = pd.read_csv(path)
        # Expect columns: 'label', 'x0', 'y0', 'z0', ..., 'x20', 'y20', 'z20'
        expected_cols = 1 + NUM_LANDMARKS * 3  # label + x,y,z for each landmark
        if df.shape[1] != expected_cols:
            print(f"Warning: Incorrect number of columns in {file_path}. Expected {expected_cols}, got {df.shape[1]}. Skipping.")
            return None
        # Ensure landmark columns are numeric
        landmark_cols = df.columns[1:]
        df[landmark_cols] = df[landmark_cols].apply(pd.to_numeric, errors='coerce')
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def load_all_gesture_data(
    data_dir: Union[str, Path], file_extension: str = ".csv"
) -> pd.DataFrame:
    """Loads gesture data from all CSV files in a directory."""
    data_path = Path(data_dir)
    all_df = []
    if not data_path.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for file_path in data_path.glob(f"*{file_extension}"):
        df = load_data_from_csv(file_path)
        if df is not None:
            all_df.append(df)

    if not all_df:
        print(f"Warning: No valid data files found in {data_dir}")
        # Return empty dataframe with expected structure if possible, or raise error
        # For simplicity, returning an empty DataFrame. Adjust if specific columns are needed.
        return pd.DataFrame()

    combined_df = pd.concat(all_df, ignore_index=True)
    return combined_df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic preprocessing: handles missing values."""
    # Drop rows with any NaN values (often indicates detection failure)
    initial_rows = len(df)
    df_cleaned = df.dropna()
    rows_dropped = initial_rows - len(df_cleaned)
    if rows_dropped > 0:
        print(f"Preprocessing: Dropped {rows_dropped} rows with missing values.")
    return df_cleaned

def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Normalizes landmarks relative to the wrist and scales them."""
    if landmarks.shape != (NUM_LANDMARKS, 3):
        raise ValueError(f"Expected landmarks shape ({NUM_LANDMARKS}, 3), got {landmarks.shape}")

    wrist_point = landmarks[WRIST].copy()
    normalized = landmarks - wrist_point # Center around wrist

    # Calculate scale factor based on distance between wrist and middle finger MCP (landmark 9)
    # This provides a relatively stable measure of hand size
    middle_mcp = landmarks[9]
    scale = np.linalg.norm(middle_mcp - wrist_point)
    if scale < 1e-6: # Avoid division by zero or near-zero
        scale = 1.0

    normalized /= scale

    # Optional: Flatten the array back to 1D if needed downstream
    # return normalized.flatten()
    return normalized # Return as (21, 3)

def calculate_relative_distances(landmarks: np.ndarray) -> np.ndarray:
    """Calculates distances between fingertips and wrist."""
    if landmarks.shape != (NUM_LANDMARKS, 3):
        raise ValueError(f"Expected landmarks shape ({NUM_LANDMARKS}, 3), got {landmarks.shape}")

    wrist_point = landmarks[WRIST]
    distances = []
    for tip_index in FINGERTIP_INDICES:
        distance = np.linalg.norm(landmarks[tip_index] - wrist_point)
        distances.append(distance)
    return np.array(distances)

def extract_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Extracts features (normalized landmarks, relative distances) and encodes labels."""
    labels = df['label'].values
    landmark_data = df.drop('label', axis=1).values

    num_samples = landmark_data.shape[0]
    if landmark_data.shape[1] != NUM_LANDMARKS * 3:
         raise ValueError(f"Expected {NUM_LANDMARKS * 3} landmark columns, got {landmark_data.shape[1]}")

    all_normalized_landmarks = []
    all_relative_distances = []

    for i in range(num_samples):
        # Reshape flat data into (21, 3)
        landmarks_3d = landmark_data[i].reshape(NUM_LANDMARKS, 3)

        # Normalize
        normalized = normalize_landmarks(landmarks_3d)
        all_normalized_landmarks.append(normalized.flatten()) # Flatten for feature vector

        # Calculate relative distances
        distances = calculate_relative_distances(normalized) # Use normalized landmarks for distances
        all_relative_distances.append(distances)

    # Combine features
    # Here we use both normalized landmark coordinates and relative distances
    # Adjust which features to use based on model performance
    feature_matrix = np.hstack((
        np.array(all_normalized_landmarks),
        np.array(all_relative_distances)
    ))

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    return feature_matrix, encoded_labels, label_encoder

def scale_features(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Scales features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    validation_size: float = 0.1,
    random_state: Optional[int] = 42,
    stratify: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Splits data into training, validation, and test sets."""
    if not (0 < test_size < 1):
        raise ValueError("test_size must be between 0 and 1")
    if not (0 <= validation_size < 1):
        raise ValueError("validation_size must be between 0 and 1")
    if test_size + validation_size >= 1.0:
         raise ValueError("The sum of test_size and validation_size must be less than 1")

    stratify_param = y if stratify else None

    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )

    # Calculate validation size relative to the remaining data
    if validation_size > 0:
        relative_val_size = validation_size / (1.0 - test_size)
        if relative_val_size >= 1.0:
             # This case should ideally be caught by the initial check, but added for safety
             raise ValueError("Validation size too large relative to remaining data after test split")

        stratify_param_val = y_temp if stratify else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=relative_val_size,
            random_state=random_state,
            stratify=stratify_param_val
        )
    else:
        # No validation set needed
        X_train, y_train = X_temp, y_temp
        # Create empty arrays for validation set to maintain return type consistency
        X_val, y_val = np.array([]).reshape(0, X.shape[1]), np.array([])


    print(f"Data split:")
    print(f"  Train: {X_train.shape[0]} samples")
    if validation_size > 0:
        print(f"  Validation: {X_val.shape[0]} samples")
    else:
        print(f"  Validation: 0 samples")
    print(f"  Test: {X_test.shape[0]} samples")

    return X_train, X_val, X_test, y_train, y_val, y_test


def load_and_prepare_data(
    data_dir: Union[str, Path],
    test_size: float = 0.2,
    validation_size: float = 0.1,
    random_state: Optional[int] = 42,
    scale: bool = True,
    stratify: bool = True
) -> dict:
    """Main function to load, preprocess, feature engineer, split, and scale data."""
    print(f"Loading data from: {data_dir}")
    df = load_all_gesture_data(data_dir)

    if df.empty:
        raise ValueError("No data loaded. Check data directory and file format.")

    print(f"Loaded {len(df)} total samples.")

    df_processed = preprocess_data(df)
    if df_processed.empty:
        raise ValueError("All data removed during preprocessing. Check data quality.")

    print(f"Preprocessing complete. {len(df_processed)} samples remaining.")

    X, y, label_encoder = extract_features(df_processed)
    print(f"Feature extraction complete. Feature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Found classes: {label_encoder.classes_}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y,
        test_size=test_size,
        validation_size=validation_size,
        random_state=random_state,
        stratify=stratify
    )

    scaler = None
    if scale:
        print("Scaling features...")
        X_train, X_val, X_test, scaler = scale_features(X_train, X_val, X_test)
        print("Scaling complete.")

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "label_encoder": label_encoder,
        "scaler": scaler, # Can be None if scale=False
        "feature_names": [f"norm_lm_{i}" for i in range(NUM_LANDMARKS * 3)] + [f"dist_tip_{j}_wrist" for j in FINGERTIP_INDICES]
    }


# Example usage (optional, can be removed or placed under if __name__ == "__main__":)
if __name__ == '__main__':
    # Create dummy data for demonstration if no real data exists
    DUMMY_DATA_DIR = Path("./dummy_gesture_data")
    NUM_SAMPLES_PER_CLASS = 50
    CLASSES = ["fist", "open_palm", "pointing"]

    if not DUMMY_DATA_DIR.exists():
        print("Creating dummy data directory...")
        DUMMY_DATA_DIR.mkdir(parents=True, exist_ok=True)

        for gesture_label in CLASSES:
            file_path = DUMMY_DATA_DIR / f"{gesture_label}_data.csv"
            if not file_path.exists():
                print(f"Creating dummy file: {file_path}")
                # Generate random landmark data (not realistic, just for structure)
                num_features = NUM_LANDMARKS * 3
                data = np.random.rand(NUM_SAMPLES_PER_CLASS, num_features) * 100 # Scaled random data
                labels = [gesture_label] * NUM_SAMPLES_PER_CLASS

                cols = ['label'] + [f'{axis}{i}' for i in range(NUM_LANDMARKS) for axis in ['x', 'y', 'z']]
                df_dummy = pd.DataFrame(data, columns=cols[1:])
                df_dummy.insert(0, 'label', labels)
                df_dummy.to_csv(file_path, index=False)
            else:
                print(f"Dummy file already exists: {file_path}")

    try:
        # Set the path to your actual data directory or use the dummy one
        DATA_DIRECTORY = DUMMY_DATA_DIR # Or replace with your actual path
        prepared_data = load_and_prepare_data(DATA_DIRECTORY, validation_size=0.15, test_size=0.15)

        print("\nData loading and preparation successful.")
        print(f"Train features shape: {prepared_data['X_train'].shape}")
        print(f"Train labels shape: {prepared_data['y_train'].shape}")
        print(f"Validation features shape: {prepared_data['X_val'].shape}")
        print(f"Validation labels shape: {prepared_data['y_val'].shape}")
        print(f"Test features shape: {prepared_data['X_test'].shape}")
        print(f"Test labels shape: {prepared_data['y_test'].shape}")
        print(f"Label Encoder Classes: {prepared_data['label_encoder'].classes_}")
        if prepared_data['scaler']:
            print(f"Scaler fitted: Yes")
        else:
            print(f"Scaler fitted: No")

        # Example: Accessing data
        # X_train_data = prepared_data['X_train']
        # y_train_labels = prepared_data['y_train']
        # class_names = prepared_data['label_encoder'].classes_

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure the data directory exists and contains valid CSV files.")
    except ValueError as e:
        print(f"\nError: {e}")
        print("Please check the data format and quality.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")