```python
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

NUM_LANDMARKS = 21
NUM_DIMENSIONS = 2 # Using x, y coordinates
MODEL_FILENAME = 'gesture_model.pkl'
LABEL_ENCODER_FILENAME = 'label_encoder.pkl'
DEFAULT_MODEL_DIR = 'trained_model'

def preprocess_landmarks(landmarks_list):
    """
    Preprocesses a list of landmark sets (one per hand detected).
    Normalizes landmarks relative to the wrist (landmark 0) and flattens them.
    Returns a list of processed feature vectors, one for each hand.
    """
    processed_data = []
    if not landmarks_list:
        return processed_data

    for landmarks in landmarks_list:
        if not landmarks or len(landmarks) != NUM_LANDMARKS:
            logging.warning(f"Invalid landmark data received: length {len(landmarks) if landmarks else 0}")
            continue # Skip invalid landmark sets

        try:
            # Ensure landmarks have x and y attributes
            landmarks_np = np.array([[lm.x, lm.y] for lm in landmarks]) # Use only x, y
        except AttributeError:
            logging.warning("Landmark object does not have 'x' or 'y' attribute.")
            continue

        # Use wrist (landmark 0) as the reference point
        wrist = landmarks_np[0]
        relative_landmarks = landmarks_np[1:] - wrist # Exclude wrist itself

        # Normalize distances by scaling relative to the maximum distance from the wrist
        max_dist = np.max(np.linalg.norm(relative_landmarks, axis=1))
        if max_dist > 1e-6: # Avoid division by zero or near-zero
             normalized_landmarks = relative_landmarks / max_dist
        else:
             normalized_landmarks = relative_landmarks # Keep as is if max_dist is too small

        # Flatten the array
        feature_vector = normalized_landmarks.flatten()

        # Expected feature vector length: (NUM_LANDMARKS - 1) * NUM_DIMENSIONS
        expected_len = (NUM_LANDMARKS - 1) * NUM_DIMENSIONS
        if len(feature_vector) == expected_len:
            processed_data.append(feature_vector)
        else:
             logging.warning(f"Feature vector length mismatch: expected {expected_len}, got {len(feature_vector)}")

    return processed_data


def train_gesture_model(X, y, model_dir=DEFAULT_MODEL_DIR, test_size=0.2, random_state=42):
    """
    Trains a gesture recognition model using RandomForestClassifier and GridSearchCV.

    Args:
        X (np.ndarray): Feature data (processed landmarks).
        y (np.ndarray): Labels corresponding to the features.
        model_dir (str): Directory to save the trained model and label encoder.
        test_size (float): Proportion of data to use for the test set.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: (trained_model, label_encoder, accuracy_on_test_set)
               Returns (None, None, 0.0) if training fails.
    """
    if X is None or y is None or len(X) == 0 or len(y) == 0:
        logging.error("Training data (X or y) is empty or None.")
        return None, None, 0.0
    if len(X) != len(y):
        logging.error(f"Mismatch between number of samples in X ({len(X)}) and y ({len(y)}).")
        return None, None, 0.0

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    logging.info(f"Training with {len(X)} samples and {num_classes} classes: {label_encoder.classes_}")

    if len(X) < 5 * num_classes: # Basic check for sufficient data per class
         logging.warning(f"Low amount of training data ({len(X)} samples) for {num_classes} classes. Model performance might be poor.")

    # Split data
    try:
        # Stratify ensures class distribution is similar in train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )
    except ValueError as e:
         logging.warning(f"Could not perform stratified split, possibly due to insufficient samples for some classes. Falling back to non-stratified split. Error: {e}")
         try:
            # Fallback to non-stratified split if stratification fails
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=random_state
            )
         except Exception as split_err:
             logging.error(f"Failed to split data: {split_err}")
             return None, None, 0.0


    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 3, 5],
        'class_weight': ['balanced', None] # Consider class imbalance
    }

    rf = RandomForestClassifier(random_state=random_state)

    # Determine appropriate number of CV folds, ensuring it's less than the smallest class count in the training set
    min_class_count = np.min(np.bincount(y_train)) if len(y_train) > 0 else 0
    cv_folds = min(5, min_class_count) if min_class_count > 1 else 3 # Default to 3 if classes are very small or only 1 sample

    if len(np.unique(y_train)) < cv_folds:
        cv_folds = len(np.unique(y_train)) # Adjust CV folds if fewer unique classes than folds

    model = None
    if cv_folds < 2:
        logging.warning("Cannot perform cross-validation with less than 2 folds/classes in training data. Training with default parameters.")
        # Fallback: Train without GridSearchCV if CV is not possible
        try:
            model = RandomForestClassifier(random_state=random_state, class_weight='balanced', n_estimators=100)
            model.fit(X_train, y_train)
        except Exception as train_err:
            logging.error(f"Failed to train model even without CV: {train_err}")
            return None, None, 0.0
    else:
        try:
            logging.info(f"Starting GridSearchCV with {cv_folds} folds...")
            grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv_folds, n_jobs=-1, verbose=1, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            logging.info(f"Best parameters found: {grid_search.best_params_}")
            logging.info(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
        except Exception as e:
            logging.error(f"GridSearchCV failed: {e}. Training with default parameters as fallback.")
            # Fallback to default