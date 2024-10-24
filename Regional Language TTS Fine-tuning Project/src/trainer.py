import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
from typing import Dict, Any

class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.writer = SummaryWriter(config.TENSORBOARD_DIR)
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        
    def train(self, train_dataset, valid_dataset):
        """Main training loop."""
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True
        )
        
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False
        )
        
        # Train the model
        self.model.train(train_loader, valid_loader)
        
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to tensorboard."""
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)