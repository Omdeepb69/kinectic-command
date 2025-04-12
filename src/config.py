import os
import pathlib
import torch

class Config:
    """
    Configuration settings for Kinectic Command.
    """

    class Paths:
        """
        Path configurations.
        """
        _BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
        DATA_DIR = _BASE_DIR / "data"
        RAW_DATA_DIR = DATA_DIR / "raw"
        PROCESSED_DATA_DIR = DATA_DIR / "processed"
        MODEL_DIR = _BASE_DIR / "models"
        LOG_DIR = _BASE_DIR / "logs"
        OUTPUT_DIR = _BASE_DIR / "output"
        CHECKPOINT_DIR = MODEL_DIR / "checkpoints"

        # Ensure directories exist
        for path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, LOG_DIR, OUTPUT_DIR, CHECKPOINT_DIR]:
            path.mkdir(parents=True, exist_ok=True)

    class Model:
        """
        Model parameters.
        """
        MODEL_NAME: str = "kinetic_transformer_v1"
        INPUT_FEATURES: int = 66 # Example: 33 landmarks * 2 (x, y)
        SEQUENCE_LENGTH: int = 50 # Example: 50 frames
        NUM_CLASSES: int = 10 # Example: 10 different gestures
        EMBEDDING_DIM: int = 128
        NUM_HEADS: int = 8
        NUM_ENCODER_LAYERS: int = 6
        DIM_FEEDFORWARD: int = 512
        DROPOUT_RATE: float = 0.1

    class Training:
        """
        Training parameters.
        """
        BATCH_SIZE: int = 64
        NUM_EPOCHS: int = 100
        LEARNING_RATE: float = 1e-4
        WEIGHT_DECAY: float = 1e-5
        OPTIMIZER: str = "AdamW" # Options: "Adam", "AdamW", "SGD"
        LOSS_FUNCTION: str = "CrossEntropyLoss"
        LR_SCHEDULER: str = "ReduceLROnPlateau" # Options: "StepLR", "ReduceLROnPlateau", None
        LR_SCHEDULER_PATIENCE: int = 10
        LR_SCHEDULER_FACTOR: float = 0.1
        EARLY_STOPPING_PATIENCE: int = 15
        GRADIENT_CLIP_VAL: float = 1.0
        VALIDATION_SPLIT: float = 0.15
        TEST_SPLIT: float = 0.15

    class Environment:
        """
        Environment configuration.
        """
        DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
        RANDOM_SEED: int = 42
        NUM_WORKERS: int = os.cpu_count() // 2 if os.cpu_count() else 1
        LOG_INTERVAL: int = 10 # Log every N batches
        SAVE_CHECKPOINT_EPOCHS: int = 5 # Save checkpoint every N epochs

# Instantiate config object for easy access
config = Config()

# Example of how to access configuration values:
# from config import config
# print(config.Paths.DATA_DIR)
# print(config.Training.BATCH_SIZE)
# print(config.Environment.DEVICE)

if __name__ == "__main__":
    # Example usage or verification print statements
    print(f"Base Directory: {config.Paths._BASE_DIR}")
    print(f"Data Directory: {config.Paths.DATA_DIR}")
    print(f"Model Directory: {config.Paths.MODEL_DIR}")
    print(f"Log Directory: {config.Paths.LOG_DIR}")
    print(f"Output Directory: {config.Paths.OUTPUT_DIR}")
    print("-" * 20)
    print(f"Model Name: {config.Model.MODEL_NAME}")
    print(f"Input Features: {config.Model.INPUT_FEATURES}")
    print(f"Num Classes: {config.Model.NUM_CLASSES}")
    print("-" * 20)
    print(f"Batch Size: {config.Training.BATCH_SIZE}")
    print(f"Learning Rate: {config.Training.LEARNING_RATE}")
    print(f"Epochs: {config.Training.NUM_EPOCHS}")
    print("-" * 20)
    print(f"Device: {config.Environment.DEVICE}")
    print(f"Random Seed: {config.Environment.RANDOM_SEED}")
    print(f"Num Workers: {config.Environment.NUM_WORKERS}")