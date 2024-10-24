# Text to Speech TTS Models for English Technical Speech and Regional Languages



------------------------------------------------------------------------------------------------------------------
# TTS Fine-tuning for Technical Vocabulary

A Node.js project for fine-tuning Text-to-Speech models to better handle technical terms and programming vocabulary.

## Features

- Custom dataset creation with technical terms
- Fine-tuning pipeline for Coqui TTS models
- Evaluation tools for both objective and subjective metrics
- Support for technical vocabulary and programming terms

## Prerequisites

- Python 3.7+
- CUDA-capable GPU (recommended for training)
- Node.js 14+

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd tts-fine-tuning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
project_root/
├── data/
│   ├── raw_data/        # Raw audio and text data
│   └── processed_data/  # Processed datasets
├── models/
│   └── fine_tuned_model/# Saved model checkpoints
├── scripts/
│   ├── data_preparation.py  # Dataset creation
│   ├── fine_tuning.py      # Model training
│   └── evaluation.py       # Model evaluation
├── requirements.txt
└── README.md
```

## Usage

### 1. Data Preparation

Prepare the custom dataset with technical terms:

```bash
python scripts/data_preparation.py
```

### 2. Model Fine-tuning

Train the model on the prepared dataset:

```bash
python scripts/fine_tuning.py
```

### 3. Evaluation

Evaluate the fine-tuned model:

```bash
python scripts/evaluation.py
```

## Evaluation Metrics

The system uses both objective and subjective metrics:

- **Objective Metrics**
  - Word Error Rate (WER)
  - CUDA Performance metrics
  - Audio quality measurements

- **Subjective Metrics**
  - Naturalness
  - Clarity
  - Technical term pronunciation
  - Overall quality

## Technical Terms Coverage

The model is trained to handle common technical terms including:
- API terminology
- Programming concepts
- Framework names
- Technical abbreviations

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Coqui TTS team for the base model
- LibriSpeech dataset
- Open-source TTS community

-------------------------------------------------------------------------------------------------------------------------


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
