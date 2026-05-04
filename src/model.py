import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) architecture suitable for MNIST.
    Inputs are flattened 28x28 (784) images.
    """
    def __init__(self, input_size=784, hidden_size=400, output_size=10):
        super(MLP, self).__init__()
        # First hidden layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Second hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # Output layer
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Flatten the input tensor to (batch_size, 784)
        x = x.view(x.size(0), -1)
        # Apply ReLU activation on the first hidden layer
        x = F.relu(self.fc1(x))
        # Apply ReLU activation on the second hidden layer
        x = F.relu(self.fc2(x))
        # Output layer (raw logits, as CrossEntropyLoss will apply Softmax)
        x = self.fc3(x)
        return x
