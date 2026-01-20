# Simple PyTorch Neural Network

A minimal, clean PyTorch project demonstrating a simple feedforward neural network.

## What This Project Does

This project implements a basic neural network using PyTorch. The model is a simple feedforward network with the architecture:
- **Input Layer** → **Hidden Layer (ReLU)** → **Output Layer**

The training script generates synthetic data and trains the model to learn a regression task.

## Project Structure

```
Arnav_mini_nn/
├── model.py          # Neural network definition
├── train.py          # Training script
├── requirements.txt  # Python dependencies
├── .gitignore        # Git ignore rules
└── README.md         # This file
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the training script:**
   ```bash
   python train.py
   ```

## How It Works

- `model.py` defines `SimpleNet`, a neural network with one hidden layer
- `train.py` generates synthetic data, trains the model, and saves it to `model.pt`
- The model is trained using Mean Squared Error (MSE) loss and Adam optimizer
- Training progress is printed every 10 epochs

## Output

After training, you'll see:
- Loss values printed every 10 epochs
- A saved model file (`model.pt`) containing the trained weights

## Requirements

- Python 3.7+
- PyTorch 2.0+
- NumPy 1.24+
