import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # Recomendación original de paper SIREN
                self.linear.weight.uniform_(-1 / self.linear.in_features, 1 / self.linear.in_features)
            else:
                bound = math.sqrt(6 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


# Define a Fourier feature layer
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

        # Fourier feature layer (output is hidden_dim)
        self.fourier = FourierFeatureLayer(input_dim, hidden_dim, scale)

        # Linear layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fourier(x)  # No activation
        for layer in self.layers:
            x = self.activation_function(layer(x))
        return self.output_layer(x)


class SIREN(nn.Module):
    """A simple implementation of SIREN (Sinusoidal Representation Networks).
    Args:
        input_dim (int): Dimension of the input.
        hidden_dim (int): Dimension of the hidden layers.
        output_dim (int): Dimension of the output.
        num_layers (int): Number of layers in the network.
        activation_function (str): Activation function to use. Default is "ReLU".
        omega_0 (float): Frequency scaling factor. Default is 30.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, activation_function="ReLU", omega_0=30):
        super(SIREN, self).__init__()

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
        self.layers.append(SineLayer(input_dim, hidden_dim, is_first=True, omega_0=omega_0))

        for _ in range(num_layers - 2):
            self.layers.append(SineLayer(hidden_dim, hidden_dim, omega_0=omega_0))

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)
