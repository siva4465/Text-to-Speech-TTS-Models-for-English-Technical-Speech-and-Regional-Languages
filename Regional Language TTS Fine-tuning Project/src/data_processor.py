import os
import librosa
import soundfile as sf
import pandas as pd
from tqdm import tqdm
from typing import Tuple, List
import numpy as np

class DataProcessor:
    def __init__(self, config):
        self.config = config
        os.makedirs(self.config.PROCESSED_PATH, exist_ok=True)
        
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load and preprocess audio file."""
        audio, sr = librosa.load(audio_path, sr=self.config.SAMPLE_RATE)
        return audio, sr
    
    def process_dataset(self):
        """Process the raw dataset and prepare it for training."""
        metadata = []
        
        for audio_file in tqdm(os.listdir(self.config.DATASET_PATH)):
            if audio_file.endswith('.wav'):
                audio_path = os.path.join(self.config.DATASET_PATH, audio_file)
                try:
                    # Load and process audio
                    audio, sr = self.load_audio(audio_path)
                    
                    # Save processed audio
                    processed_path = os.path.join(self.config.PROCESSED_PATH, audio_file)
                    sf.write(processed_path, audio, sr)
                    
                    # Store metadata
                    duration = len(audio) / sr
                    metadata.append({
                        'file_name': audio_file,
                        'duration': duration,
                        'sample_rate': sr
                    })
                    
                except Exception as e:
                    print(f"Error processing {audio_file}: {str(e)}")
        
        # Save metadata
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(os.path.join(self.config.PROCESSED_PATH, 'metadata.csv'), 
                          index=False)
        
        return metadata_df