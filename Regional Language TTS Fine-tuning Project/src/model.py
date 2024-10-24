from TTS.utils.audio import AudioProcessor
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits
import torch
from typing import Dict, Any

class TTSModel:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.audio_processor = None
        
    def setup_model(self):
        """Initialize and setup the TTS model."""
        # Create model config
        model_config = VitsConfig(
            audio=BaseDatasetConfig(
                sample_rate=self.config.SAMPLE_RATE,
            )
        )
        
        # Initialize model
        self.model = Vits(model_config)
        self.model.to(self.device)
        
        # Initialize audio processor
        self.audio_processor = AudioProcessor(
            sample_rate=self.config.SAMPLE_RATE,
        )
        
    def train(self, train_loader, valid_loader):
        """Train the model."""
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE
        )
        
        for epoch in range(self.config.EPOCHS):
            self.model.train()
            train_loss = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch)
                loss = outputs['loss']
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            if epoch % self.config.EVAL_INTERVAL == 0:
                self.evaluate(valid_loader)
                
            # Save checkpoint
            self.save_checkpoint(epoch)
    
    def evaluate(self, valid_loader) -> Dict[str, float]:
        """Evaluate the model."""
        self.model.eval()
        eval_loss = 0
        
        with torch.no_grad():
            for batch in valid_loader:
                outputs = self.model(batch)
                eval_loss += outputs['loss'].item()
        
        return {'eval_loss': eval_loss / len(valid_loader)}
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint_path = f"{self.config.CHECKPOINT_DIR}/model_epoch_{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
        }, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint['epoch']
    
    def synthesize(self, text: str) -> torch.Tensor:
        """Synthesize speech from text."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.synthesize(text)
        return outputs