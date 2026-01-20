# arnav
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import SimpleNet


def generate_synthetic_data(n_samples=1000, input_size=10):
    """
    Generate a simple synthetic dataset for training.
    Creates a regression problem where y is a function of x.
    """
    # Generate random input features
    X = np.random.randn(n_samples, input_size)
    
    # Create target: sum of first 3 features + noise
    y = X[:, 0] + X[:, 1] * 0.5 + X[:, 2] * 0.3 + np.random.randn(n_samples) * 0.1
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).unsqueeze(1)
    
    return X_tensor, y_tensor


def train_model():
    """Train the neural network on synthetic data."""
    # Hyperparameters
    input_size = 10
    hidden_size = 64
    output_size = 1
    learning_rate = 0.01
    num_epochs = 100
    print_interval = 10
    
    # Generate synthetic dataset
    print("Generating synthetic dataset...")
    X, y = generate_synthetic_data(n_samples=1000, input_size=input_size)
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    
    # Initialize model, loss, and optimizer
    model = SimpleNet(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print("-" * 50)
    
    # Training loop
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print loss every few epochs
        if (epoch + 1) % print_interval == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
    
    print("-" * 50)
    print("Training completed!")
    
    # Save the trained model
    model_path = "model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train_model()
