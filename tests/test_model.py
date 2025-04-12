import pytest
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.exceptions import NotFittedError

# Attempt to import the actual model, assuming it exists in model.py
# If model.py or SimpleModel doesn't exist, these tests will fail.
try:
    from model import SimpleModel
except ImportError:
    # Define a dummy SimpleModel if the actual one cannot be imported
    # This allows the test structure to be demonstrated, but replace
    # this with the actual import 'from model import SimpleModel'
    # when model.py is available.
    # NOTE: For the final submission as per instructions, this dummy class
    # should ideally be removed, assuming 'from model import SimpleModel' works.
    # However, to make the code runnable *as is* for demonstration/testing
    # of the test file itself, it's included here.
    # If you have model.py, remove this class definition.
    class SimpleModel(BaseEstimator, RegressorMixin):
        def __init__(self, learning_rate=0.01, iterations=100):
            self.learning_rate = learning_rate
            self.iterations = iterations
            self.weights_ = None
            self.bias_ = None
            self._is_fitted = False

        def _initialize_weights(self, n_features):
            self.weights_ = np.zeros(n_features)
            self.bias_ = 0.0

        def train(self, X, y):
            X, y = check_X_y(X, y)
            n_samples, n_features = X.shape
            if self.weights_ is None or self.bias_ is None:
                 self._initialize_weights(n_features)
            for _ in range(self.iterations):
                y_pred = X.dot(self.weights_) + self.bias_
                dw = (1 / n_samples) * X.T.dot(y_pred - y)
                db = (1 / n_samples) * np.sum(y_pred - y)
                self.weights_ -= self.learning_rate * dw
                self.bias_ -= self.learning_rate * db
            self._is_fitted = True
            return self

        def predict(self, X):
            check_is_fitted(self, '_is_fitted')
            X = check_array(X)
            if X.shape[1] != self.weights_.shape[0]:
                 raise ValueError(f"Input has {X.shape[1]} features, but model expects {self.weights_.shape[0]}")
            return X.dot(self.weights_) + self.bias_

        def evaluate(self, X, y):
            check_is_fitted(self, '_is_fitted')
            X, y = check_X_y(X, y)
            y_pred = self.predict(X)
            mse = np.mean((y - y_pred) ** 2)
            return {'mse': mse}

        def get_params(self, deep=True):
            return {"learning_rate": self.learning_rate, "iterations": self.iterations}

        def set_params(self, **parameters):
            for parameter, value in parameters.items():
                setattr(self, parameter, value)
            return self


@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = np.random.rand(50, 3) * 10
    true_weights = np.array([2, 3, -1])
    true_bias = 5
    noise = np.random.randn(50) * 0.5
    y = X.dot(true_weights) + true_bias + noise
    return X, y

@pytest.fixture
def initialized_model():
    return SimpleModel(learning_rate=0.01, iterations=100)

@pytest.fixture
def trained_model(initialized_model, sample_data):
    X, y = sample_data
    return initialized_model.train(X, y)


def test_model_initialization_defaults():
    model = SimpleModel()
    assert model.learning_rate == 0.01
    assert model.iterations == 100
    assert model.weights_ is None
    assert model.bias_ is None
    assert not hasattr(model, '_is_fitted') or not model._is_fitted

def test_model_initialization_custom_params():
    model = SimpleModel(learning_rate=0.05, iterations=500)
    assert model.learning_rate == 0.05
    assert model.iterations == 500
    assert model.weights_ is None
    assert model.bias_ is None
    assert not hasattr(model, '_is_fitted') or not model._is_fitted


def test_model_train_runs(initialized_model, sample_data):
    X, y = sample_data
    try:
        initialized_model.train(X, y)
    except Exception as e:
        pytest.fail(f"model.train() raised an exception {e}")

def test_model_train_sets_attributes(initialized_model, sample_data):
    X, y = sample_data
    model = initialized_model.train(X, y)
    assert model.weights_ is not None
    assert model.bias_ is not None
    assert isinstance(model.weights_, np.ndarray)
    assert isinstance(model.bias_, (float, np.float64))
    assert model.weights_.shape == (X.shape[1],)
    assert model._is_fitted is True

def test_model_train_returns_self(initialized_model, sample_data):
    X, y = sample_data
    model = initialized_model
    returned_value = model.train(X, y)
    assert returned_value is model


def test_model_predict_runs(trained_model, sample_data):
    X, _ = sample_data
    X_test = X[:5]
    try:
        predictions = trained_model.predict(X_test)
        assert predictions is not None
    except Exception as e:
        pytest.fail(f"model.predict() raised an exception {e}")

def test_model_predict_output_shape(trained_model, sample_data):
    X, _ = sample_data
    X_test = X[:10]
    predictions = trained_model.predict(X_test)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (X_test.shape[0],)

def test_model_predict_before_train(initialized_model, sample_data):
    X, _ = sample_data
    X_test = X[:5]
    with pytest.raises((NotFittedError, AttributeError, ValueError)):
         initialized_model.predict(X_test)

def test_model_predict_incorrect_features(trained_model, sample_data):
    X, _ = sample_data
    X_test_wrong_features = np.random.rand(5, X.shape[1] + 1)
    with pytest.raises(ValueError):
        trained_model.predict(X_test_wrong_features)


def test_model_evaluate_runs(trained_model, sample_data):
    X, y = sample_data
    X_test, y_test = X[:10], y[:10]
    try:
        metrics = trained_model.evaluate(X_test, y_test)
        assert metrics is not None
    except Exception as e:
        pytest.fail(f"model.evaluate() raised an exception {e}")

def test_model_evaluate_output_format(trained_model, sample_data):
    X, y = sample_data
    X_test, y_test = X[10:20], y[10:20]
    metrics = trained_model.evaluate(X_test, y_test)
    assert isinstance(metrics, dict)
    assert 'mse' in metrics
    assert isinstance(metrics['mse'], (float, np.float64))

def test_model_evaluate_mse_non_negative(trained_model, sample_data):
    X, y = sample_data
    X_test, y_test = X[20:30], y[20:30]
    metrics = trained_model.evaluate(X_test, y_test)
    assert metrics['mse'] >= 0.0

def test_model_evaluate_before_train(initialized_model, sample_data):
    X, y = sample_data
    X_test, y_test = X[:5], y[:5]
    with pytest.raises((NotFittedError, AttributeError, ValueError)):
         initialized_model.evaluate(X_test, y_test)


def test_model_get_set_params(initialized_model):
    params = initialized_model.get_params()
    assert 'learning_rate' in params
    assert 'iterations' in params
    assert params['learning_rate'] == 0.01
    assert params['iterations'] == 100

    initialized_model.set_params(learning_rate=0.99, iterations=123)
    new_params = initialized_model.get_params()
    assert new_params['learning_rate'] == 0.99
    assert new_params['iterations'] == 123