import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(3, 2)  
        self.output = nn.Linear(2, 1) 

        # Manually setting the given weights and biases
        self.hidden.weight = nn.Parameter(torch.tensor([
            [0.5, -0.3, 0.2],  
            [0.6, -0.4, 0.1]   
        ], dtype=torch.float32))
        
        self.hidden.bias = nn.Parameter(torch.tensor([0.1, -0.2], dtype=torch.float32))

        self.output.weight = nn.Parameter(torch.tensor([[0.7, -0.5]], dtype=torch.float32))
        self.output.bias = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))  # Bias for output neuron (default 0)

    def forward(self, x):
        h = torch.sigmoid(self.hidden(x))  
        y = self.output(h)  
        return h, y

# Initialize model
model = SimpleNN()

# Define input tensor
x = torch.tensor([2.0, 3.0, 4.0])  # Input vector

# Forward pass
h_hidden, y = model(x)

# Compute neuron importance as the absolute values of the output layer weights
importance_h1 = abs(model.output.weight[0, 0]).item()
importance_h2 = abs(model.output.weight[0, 1]).item()

# Display results
print(f"Hidden layer activations: {h_hidden}")
print(f"Output y: {y.item():.4f}")
print("Neuron Importance:")
print(f"  h1: {importance_h1:.4f}")
print(f"  h2: {importance_h2:.4f}")
