# Regional Language TTS Fine-tuning Project

This project implements fine-tuning of Text-to-Speech (TTS) models for regional languages using Coqui TTS.

## Project Structure
```
.
├── src/
│   ├── config.py          # Configuration settings
│   ├── data_processor.py  # Data processing utilities
│   ├── model.py           # TTS model implementation
│   ├── trainer.py         # Training loop implementation
│   └── evaluator.py       # Model evaluation utilities
├── data/
│   ├── raw/              # Raw audio files
│   └── processed/        # Processed dataset
├── checkpoints/          # Model checkpoints
├── logs/                 # Training logs
├── requirements.txt      # Project dependencies
├── main.py              # Main entry point
└── README.md            # Project documentation
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your dataset:
   - Place your raw audio files in `data/raw/`
   - Ensure audio files are in WAV format
   - Include corresponding text transcriptions

## Usage

### Training
```bash
python main.py --train
```

### Evaluation
```bash
python main.py --evaluate
```

### Speech Synthesis
```bash
python main.py --synthesize "Your text here"
```

## Configuration

Modify `src/config.py` to adjust:
- Model parameters
- Training settings
- Dataset configuration
- Evaluation metrics

## Features

- Multi-language TTS support
- Fine-tuning capabilities
- Comprehensive evaluation metrics
- TensorBoard integration
- Checkpoint management
- Audio preprocessing utilities