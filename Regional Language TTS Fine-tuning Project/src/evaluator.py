import numpy as np
from typing import Dict, List
import pandas as pd
from sklearn.metrics import mean_squared_error
import librosa

class Evaluator:
    def __init__(self, config):
        self.config = config
        
    def calculate_mos(self, predictions: List[np.ndarray], 
                     ground_truth: List[np.ndarray]) -> float:
        """Calculate Mean Opinion Score (MOS)."""
        scores = []
        for pred, truth in zip(predictions, ground_truth):
            # Calculate various audio quality metrics
            mse = mean_squared_error(truth, pred)
            mel_pred = librosa.feature.melspectrogram(y=pred)
            mel_truth = librosa.feature.melspectrogram(y=truth)
            mel_mse = mean_squared_error(mel_truth, mel_pred)
            
            # Combine metrics into a single score
            score = 1 / (1 + mse + mel_mse)
            scores.append(score)
            
        return np.mean(scores)
    
    def evaluate_model(self, model, test_dataset) -> Dict[str, float]:
        """Evaluate model performance."""
        predictions = []
        ground_truth = []
        
        for text, audio in test_dataset:
            pred_audio = model.synthesize(text)
            predictions.append(pred_audio.numpy())
            ground_truth.append(audio.numpy())
        
        metrics = {
            'mos_score': self.calculate_mos(predictions, ground_truth),
        }
        
        return metrics
    
    def generate_report(self, metrics: Dict[str, float]) -> pd.DataFrame:
        """Generate evaluation report."""
        report = pd.DataFrame(metrics.items(), columns=['Metric', 'Value'])
        report.to_csv(f"{self.config.LOG_DIR}/evaluation_report.csv", index=False)
        return report