# MNIST Digit Classification using CNN

A convolutional neural network built in PyTorch to classify handwritten digits from the MNIST dataset, achieving **99.33% accuracy** on the held-out test set.

## Overview

This project implements an end-to-end deep learning pipeline — from data loading and preprocessing to model architecture design, training with early stopping, and evaluation — using the classic MNIST dataset of handwritten digits (0–9).

## Dataset

- **Source:** MNIST (via `torchvision.datasets`)
- **Training set:** 54,000 images
- **Validation set:** 6,000 images
- **Test set:** 10,000 images

The original 60,000-image training set was split into training and validation subsets to enable proper model selection during training.

## Model Architecture

A deep CNN with three convolutional blocks of increasing depth, followed by a fully connected classifier:

```
Input (1x28x28)
  → Conv2d(1→64, k=7) → ReLU → MaxPool2d
  → Conv2d(64→128, k=3) → ReLU
  → Conv2d(128→128, k=3) → ReLU → MaxPool2d
  → Conv2d(128→256, k=3) → ReLU
  → Conv2d(256→256, k=3) → ReLU → MaxPool2d
  → Flatten
  → Linear(2304→128) → ReLU → Dropout(0.5)
  → Linear(128→64) → ReLU → Dropout(0.5)
  → Linear(64→10)
```

- **Convolutions:** "same" padding to preserve spatial dimensions within blocks
- **Regularization:** Dropout (p=0.5) in the fully connected layers to reduce overfitting
- **Pooling:** Max pooling after each convolutional block to progressively downsample

## Training

- **Optimizer:** AdamW
- **Loss function:** Cross-entropy loss
- **Metric:** Multiclass accuracy (via `torchmetrics`)
- **Batch size:** 32
- **Max epochs:** 40
- **Early stopping:** Training halts if validation accuracy doesn't improve for 10 consecutive epochs (patience-based), and the best-performing model weights are automatically restored at the end of training

Training converged and triggered early stopping at epoch 29, with the best validation accuracy reaching **99.32%**.

## Results

| Metric | Score |
|---|---|
| Best validation accuracy | 99.32% |
| **Test accuracy** | **99.33%** |

## Tech Stack

- Python
- PyTorch
- Torchvision
- TorchMetrics

## Usage

1. Install dependencies:
   ```bash
   pip install torch torchvision torchmetrics
   ```
2. Run the notebook `MNIST_CNN.ipynb` cell by cell. The MNIST dataset will be automatically downloaded to a local `./data` directory on first run.
3. GPU (CUDA) is used by default for training and evaluation — modify `.to("cuda")` calls to `.to("cpu")` if running without a GPU.

## Possible Extensions

- Add data augmentation (rotation, translation, elastic distortions) to improve robustness
- Experiment with batch normalization between convolutional layers
- Try a lighter architecture to compare accuracy/efficiency trade-offs
- Visualize misclassified examples and learned filters
