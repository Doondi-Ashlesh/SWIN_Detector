# Detection of deepfakes using SWIN Transformer
## Description

This project implements an end-to-end image classification pipeline for detecting real and manipulated (deepfake) images using the Swin Transformer (`swin-tiny`) architecture from Hugging Face Transformers. It includes dataset preparation, training, evaluation, and a simple Gradio-based inference interface.

The model is trained to classify images into multiple categories including:
- `real`
- `Deepfakes`
- `Face2Face`
- `FaceSwap`
- `NeuralTextures`

## Project Structure

```
├── image_extractor.py             # Extracts frames from video datasets and creates train/test splits
├── swin-tiny-complete-training.py# Trains Swin Transformer on processed image dataset
├── model-testing.py              # Evaluates the saved Swin model on the test set
├── gradio-test.py                # A simple Gradio interface demo (placeholder)
├── requirements.txt              # Required Python dependencies
├── models/                       # Saved trained model (after training)
├── data/                         # Train/test image data folders created from extractor
├── cache/                        # Cache for Hugging Face datasets
```

## Features

- **Frame Extraction**: Convert deepfake videos into frames and split into train/test.
- **Multi-class Classification**: Classifies real vs various types of manipulated media.
- **Transfer Learning**: Fine-tunes Swin-Tiny Transformer using Hugging Face's Trainer.
- **Evaluation Metrics**: Computes F1 Score, Precision, Recall, and Accuracy.
- **Gradio UI**: Includes a basic interactive web interface (demo placeholder).

## Dataset Format

This project assumes a Deepfake Detection dataset structure similar to:

```
dataset/
├── original_sequences/           # Videos of real individuals
├── manipulated_sequences/
│   ├── Deepfakes/
│   ├── Face2Face/
│   ├── FaceSwap/
│   └── NeuralTextures/
```

## Usage

### 1. Frame Extraction & Dataset Preparation

```bash
python image_extractor.py
```

- Extracts frames from `.mp4` files
- Resizes to 224x224
- Saves to `data/train` and `data/test` folders (80:20 split)

### 2. Train the Swin Transformer

```bash
python swin-tiny-complete-training.py
```

- Uses Hugging Face's `Trainer`
- Loads pre-trained `microsoft/swin-tiny-patch4-window7-224`
- Saves model and metrics to `./models/` and `./results/`

### 3. Test the Trained Model

```bash
python model-testing.py
```

- Loads the trained model from `./models/`
- Evaluates on `data/test`
- Reports Accuracy, F1, Precision, Recall

### 4. Launch Gradio UI (Demo)

```bash
python gradio-test.py
```

- A basic "Hello, name!" demo using Gradio.
- Replace with an image classifier interface if needed.

## Requirements

Install required dependencies:

```bash
pip install -r requirements.txt
```

**Requirements include:**
- PyTorch
- Hugging Face `transformers`, `datasets`
- OpenCV
- Gradio
- `evaluate` for metric computation

## Model Details

- **Architecture**: Swin-Tiny Transformer
- **Input Size**: 224 x 224 RGB images
- **Training Strategy**: Epoch-based, gradient accumulation, learning rate warmup
- **Evaluation**: Runs at each epoch end, logs best model by accuracy

## Metrics Logged

- **F1 Score (macro)**
- **Precision**
- **Recall**
- **Accuracy**

Metrics are saved under:
- `results/swin-tiny-complete/`
- `models/swin-tiny-complete/`

## Notes

- All video decoding, resizing, and augmentation handled with OpenCV and Hugging Face APIs.
- Gradio UI is a placeholder and should be extended for real image classification demos.
- You can cache Hugging Face datasets locally using the `./cache/` directory.

## 🔒 License

This project is for academic or research purposes only. Please ensure you have the right to use the dataset you provide.
