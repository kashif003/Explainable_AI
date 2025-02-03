import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Define the neural network as a PyTorch module
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Hidden layer (2 neurons)
        self.fc1 = nn.Linear(3, 2, bias=True)  
        # Output layer (1 neuron)
        self.fc2 = nn.Linear(2, 1, bias=True)  

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))  # Hidden layer activation
        x = torch.sigmoid(self.fc2(x))  # Output layer activation
        return x
# Initialize the model
model = SimpleNN()


# Manually set the weights and biases
with torch.no_grad():
    model.fc1.weight = nn.Parameter(torch.tensor([[2.0, -1.0, 0.5], [-1.0, 1.5, -0.5]]))  # w11, w12, w13 | w21, w22, w23
    model.fc1.bias = nn.Parameter(torch.tensor([0.5, -0.5]))  # b1, b2

    model.fc2.weight = nn.Parameter(torch.tensor([[1.0, -2.0]]))  # wo1, wo2
    model.fc2.bias = nn.Parameter(torch.tensor([0.5]))  # bo

# Input tensor with requires_grad=True for saliency computation
input_tensor = torch.tensor([1, 2.0, -1.0], requires_grad=True)
# Forward pass
output = model(input_tensor)
# Compute gradients (saliency map)
output.backward()
# Get saliency map (absolute value of gradients)
saliency = input_tensor.grad.abs()
feature_names = [r'$x_1$', r'$x_2$', r'$x_3$']  # Labels for input features

plt.figure(figsize=(6, 4))
plt.bar(feature_names, saliency.detach().numpy())
plt.xlabel("Input Features")
plt.ylabel("Saliency (Gradient Magnitude)")
plt.title("Saliency Map: Input Importance")
plt.ylim(0, max(saliency.detach().numpy()) + 0.05)  # Set y-axis limit for better visualization
plt.show()

# Print results
print(f"Output (y) = {output.item():.4f}")
print(f"Saliency Map (dy/dx):")
print(f"dy/dx1 = {saliency[0].item():.4f}")
print(f"dy/dx2 = {saliency[1].item():.4f}")
print(f"dy/dx3 = {saliency[2].item():.4f}")