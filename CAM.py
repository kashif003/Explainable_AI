import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Define a simple CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, bias=False)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(4, 2, bias=False)  

    def forward(self, x):
        feature_map = self.conv(x)
        feature_map_relu = self.relu(feature_map)
        flattened = feature_map_relu.view(-1)
        scores = self.fc(flattened)
        return feature_map_relu, scores

# Initialize Model
model = CNNModel()

# Set weights manually
model.conv.weight.data = torch.tensor([[[[0.1, 0.2], [0.3, 0.4]]]])
model.fc.weight.data = torch.tensor([[0.5, 0.2, 0.1, 0.3], [0.4, 0.6, 0.2, 0.5]])

# Define Input Image (3x3 Grayscale)
I = torch.tensor([[0.4, 0.6, 0.3],
                  [0.1, 0.5, 0.2],
                  [0.7, 0.6, 0.9]]).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

# Forward Pass
feature_map_relu, scores = model(I)

# Compute Class Activation Maps (CAM)
w = model.fc.weight.view(2, 2, 2)
CAM_1 = feature_map_relu * w[0]
CAM_2 = feature_map_relu * w[1]

# Define color mapping
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['green', 'yellow', 'red'])

# Normalize values to range [0, 1]
def normalize_cam(cam):
    cam = cam.detach().numpy()
    return (cam - np.min(cam)) / (np.max(cam) - np.min(cam))

# Plot Class Activation Maps
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(normalize_cam(CAM_1.squeeze()), cmap=cmap, interpolation='nearest')
axes[0].set_title("Class 1 Activation Map")
axes[1].imshow(normalize_cam(CAM_2.squeeze()), cmap=cmap, interpolation='nearest')
axes[1].set_title("Class 2 Activation Map")
plt.show()

# Print Results
print("Feature Map After Convolution:")
print(feature_map_relu.squeeze())
print("\nClass 1 Activation Map:")
print(CAM_1.squeeze())
print("\nClass 2 Activation Map:")
print(CAM_2.squeeze())

# Softmax to Get Prediction Probabilities
pred_probs = F.softmax(scores, dim=0)
print("\nSoftmax Class Probabilities:")
print(pred_probs.squeeze())

# Prediction
pred_class = torch.argmax(pred_probs).item()
print("\nPredicted Class:", pred_class + 1)  # +1 for 1-based indexing
