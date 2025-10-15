# Assignment 1: Multiple Layer Neural Networks

## Overview

This assignment explores building and training multi-layer neural networks for image classification on the MNIST handwritten digit dataset.

## Key Components

### 1. Multi-Layer Model Architecture
- Implemented models with 2-5 affine transformation layers
- Explored different activation functions (Sigmoid, Tanh, ReLU)
- Used 100-dimensional hidden layers between input (784) and output (10) layers
- Final layer outputs class logits for 10 digit classes (0-9)

### 2. Training Methodology
- **Loss Function**: Cross-entropy loss (combined with softmax for efficiency)
- **Optimization**: Gradient descent with backpropagation
- **Dataset Split**:
  - Training set: 48,000 samples (80%)
  - Validation set: 12,000 samples (20%)
  - Test set: 10,000 samples

### 3. Key Investigations
- Effect of model depth on training and validation performance
- Sensitivity to parameter initialization scales
- Comparison of activation functions (Sigmoid vs. Tanh vs. ReLU)
- Convergence speed and training stability across architectures

### 4. Implementation Approaches
- **Custom Framework**: Built from scratch using NumPy (mlp module)
- **PyTorch**: Modern deep learning framework comparison
- Implemented forward propagation (fprop) and backpropagation (bprop) methods

## Results

Achieved ~95-97% test accuracy on MNIST with optimized multi-layer architectures, demonstrating the effectiveness of deeper networks with appropriate hyperparameter tuning.

## Files

- [03_Multiple_layer_models.ipynb](../../notebooks/03_Multiple_layer_models.ipynb): Main notebook with experiments
- [03b_extended_pytorch_tests.ipynb](../../notebooks/03b_extended_pytorch_tests.ipynb): Extended PyTorch implementations
