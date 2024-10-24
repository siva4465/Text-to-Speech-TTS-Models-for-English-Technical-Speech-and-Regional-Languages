class Config:
    # Model Configuration
    MODEL_NAME = "tts_models/multilingual/multi-dataset/your_tts"
    LANGUAGE = "hi"  # Change this to your target language code
    
    # Dataset Configuration
    DATASET_PATH = "data/raw"
    PROCESSED_PATH = "data/processed"
    SAMPLE_RATE = 22050
    
    # Training Configuration
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    CHECKPOINT_DIR = "checkpoints"
    
    # Evaluation Configuration
    TEST_SIZE = 0.2
    EVAL_INTERVAL = 1000
    
    # Logging Configuration
    LOG_DIR = "logs"
    TENSORBOARD_DIR = "tensorboard_logs"