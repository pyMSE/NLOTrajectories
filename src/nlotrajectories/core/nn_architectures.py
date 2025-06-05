import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


#Define a Fourier feature layer
class FourierFeatureLayer(nn.Module):
    def __init__(self, in_features, out_features, scale=1.0):
        super(FourierFeatureLayer, self).__init__()
        self.scale = scale
        self.weights = nn.Parameter(torch.randn(in_features, out_features) * scale)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return torch.cos(x @ self.weights + self.bias) * self.scale
    
# Define a simple MLP with Fourier features
class FourierMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, scale=1.0, activation_function="ReLU"):
        super(FourierMLP, self).__init__()
        
        # Map string to activation function
        activation_functions = {
            "ReLU": F.relu,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
            "leaky_relu": F.leaky_relu,
        }
        if activation_function not in activation_functions:
            raise ValueError(f"Unsupported activation function: {activation_function}")
        
        self.activation_function = activation_functions[activation_function]
        self.layers = nn.ModuleList()
        self.layers.append(FourierFeatureLayer(input_dim, hidden_dim, scale))
        
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = self.activation_function(layer(x))
        return self.output_layer(x)