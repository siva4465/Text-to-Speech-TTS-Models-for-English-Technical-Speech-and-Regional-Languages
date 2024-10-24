import argparse
from src.config import Config
from src.data_processor import DataProcessor
from src.model import TTSModel
from src.trainer import Trainer
from src.evaluator import Evaluator
import torch
from torch.utils.data import random_split

def main():
    parser = argparse.ArgumentParser(description='Regional Language TTS Training')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--synthesize', type=str, help='Synthesize speech from text')
    args = parser.parse_args()
    
    # Initialize configuration
    config = Config()
    
    # Initialize components
    data_processor = DataProcessor(config)
    model = TTSModel(config)
    trainer = Trainer(model, config)
    evaluator = Evaluator(config)
    
    if args.train:
        # Process dataset
        metadata_df = data_processor.process_dataset()
        
        # Prepare dataset
        dataset_size = len(metadata_df)
        train_size = int((1 - config.TEST_SIZE) * dataset_size)
        test_size = dataset_size - train_size
        
        train_dataset, test_dataset = random_split(
            metadata_df,
            [train_size, test_size]
        )
        
        # Train model
        trainer.train(train_dataset, test_dataset)
        
    if args.evaluate:
        # Load best model
        model.load_checkpoint(f"{config.CHECKPOINT_DIR}/best_model.pth")
        
        # Evaluate model
        metrics = evaluator.evaluate_model(model, test_dataset)
        report = evaluator.generate_report(metrics)
        print("\nEvaluation Report:")
        print(report)
        
    if args.synthesize:
        # Load best model
        model.load_checkpoint(f"{config.CHECKPOINT_DIR}/best_model.pth")
        
        # Synthesize speech
        audio = model.synthesize(args.synthesize)
        print(f"Speech synthesized and saved to: {config.PROCESSED_PATH}/output.wav")

if __name__ == "__main__":
    main()