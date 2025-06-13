import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd.functional import jacobian
import os

class DirectionVectorAnalyzer:
    def __init__(self, model: nn.Module):
        """
        Initialize the analyzer.
        :param model: Input neural network model (PyTorch Module)
        """
        self.model = model.eval()
        self._zero_bias()
        self.layers = self._parse_model_layers()

    def _parse_model_layers(self) -> list:
        """Parse model layers to extract parameters of each neural network layer"""
        layers = []
        for name, layer in self.model.named_children():
            layers.append((name, layer))
        return layers

    def _zero_bias(self):
        """Set biases to zero for all linear and batch normalization layers in the model"""
        for layer in self.model.modules():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.BatchNorm1d):
                if layer.bias is not None:
                    layer.bias.data.zero_()

    def get_network_structure(self):
        """
        Get the shape of each layer in the neural network
        """
        print(self.model)
        return 0

    def get_weight_vector(
        self, 
        layer_idx: int, 
        neuron_idx
    ) -> torch.Tensor:
        """
        Extract the weight vector of a specified neuron in a specified layer.
        :param layer_idx: Index of the layer (the indexed layer must be a fully connected layer)
        :param neuron_idx: Index of the neuron in the layer (can be a single integer or a list of integers)
        :return: Weight vector (shape: [number of neurons in the next layer])
        """
        if not isinstance(self.model[layer_idx], nn.Linear):
            raise IndexError("The indexed layer must be a fully connected layer")
        if isinstance(neuron_idx, int):
            neuron_idx = [neuron_idx]
        elif not isinstance(neuron_idx, list):
            raise TypeError("neuron_idx must be an integer or a list of integers")
        
        with torch.no_grad():
            layer_name, layer = self.layers[layer_idx]
            weight = layer.weight  # shape: [number of neurons in the current layer, number of neurons in the next layer]
            df = pd.DataFrame(weight.detach().numpy()) 
            input_data = df.T
            input_data = input_data
            input_data = input_data.loc[neuron_idx,:]
            input_data = torch.tensor(input_data.values, dtype=torch.float32)

        return input_data

    def forward_propagate(
        self, 
        weight_vector: torch.Tensor, 
        start_layer_idx: int,
        elongation_ratio: int = 10000
    ) -> torch.Tensor:
        """
        Propagate the weight vector forward through the network to the output layer.
        :param weight_vector: Input weight vector (must match the dimension of the next layer of start_layer)
        :param start_layer_idx: Index to start propagation from (in the order of the neural network)
        :param elongation_ratio: Scaling factor to proportionally extend the weight vector (default: 10000 times magnification)
        :return: Output layer result (shape: [number of output layer neurons])
        """
        device = next(self.model.parameters()).device
    
        with torch.no_grad():
            specific_layer = self.model[start_layer_idx:]
            output = specific_layer(weight_vector*elongation_ratio)
            output = pd.DataFrame(output.detach().numpy())
            return output

    def get_jacobian(
        self, 
        layer_idx: int, 
        neuron_idx: int
    ) -> torch.Tensor:
        """
        Compute the Jacobian matrix of a specified neuron in a specified layer with respect to the output layer neurons.
        :param layer_idx: Index of the layer (in the order of fully connected layers)
        :param neuron_idx: Index of the neuron in the layer
        :return: Jacobian matrix (shape: [number of output layer neurons, number of current layer neurons])
        """
        if layer_idx >= len(self.layers):
            raise IndexError("Layer index out of range")
        layer_name, layer = self.layers[layer_idx]
        num_inputs = layer.in_features

        def partial_model(input):
            # Create an input where only the specified neuron has the input value, others are zero
            input_tensor = torch.zeros(num_inputs, dtype=torch.float32)
            input_tensor[neuron_idx] = input
            input_tensor = input_tensor.unsqueeze(0)
            return self.model(input_tensor).squeeze(0)

        jac = jacobian(partial_model, torch.tensor(0.0))
        return jac

    def compute_jacobian_at_sample(
        self,
        sample_data: torch.Tensor,  # Input sample point (must include batch dimension, shape: [1, input_dim])
        target_layer: nn.Module,     # Target layer (e.g., model.fc2)
        output_layer: nn.Module = None  # Target output layer (default: final output of the entire model)
    ) -> torch.Tensor:
        """
        Compute the Jacobian matrix of the target layer's input neurons with respect to the output layer at a specified sample point.
        :param sample_data: Input sample point, must include batch dimension (shape: [1, input_dim]).
        :param target_layer: Target layer for which to compute the Jacobian matrix of input neurons with respect to the output layer.
        :param output_layer: Target output layer for Jacobian computation (default: None, uses the final output of the entire model).
        :return: Computed Jacobian matrix (shape: [output_dim, target_input_dim]).
        """
        device = next(self.model.parameters()).device
        sample_data = sample_data.to(device).requires_grad_(True)
        
        # Find the index of the target layer in named_children()
        target_layer_idx = None
        for idx, (layer_name, layer_obj) in enumerate(self.layers):
            if layer_obj is target_layer:
                target_layer_idx = idx
                break
        if target_layer_idx is None:
            raise ValueError("The target layer was not found in the model's child layers")
        
        # Define truncated forward propagation (start computing output from the target layer)
        def truncated_forward(x):
            # Forward propagate through all layers before the target layer
            for layer_name, layer_obj in self.layers[:target_layer_idx]:
                x = layer_obj(x)
            return x

        def forward(x):    
            # Continue propagation to the specified output layer or the end of the model
            if output_layer is not None:
                found_output = False
                for layer_name, layer_obj in self.layers[target_layer_idx:]:
                    x = layer_obj(x)
                    if layer_obj is output_layer:
                        found_output = True
                        break
                if not found_output:
                    raise ValueError("The specified output_layer was not found after the target layer")
            else:
                for layer_name, layer_obj in self.layers[target_layer_idx:]:
                    x = layer_obj(x)
            return x  # Final output
        
        # Compute Jacobian matrix (shape: [output_dim, target_input_dim])
        jac = torch.autograd.functional.jacobian(forward, truncated_forward(sample_data), vectorize=True)

        # Get the input dimension of the target layer
        if isinstance(target_layer, nn.Linear):
            target_input_dim = target_layer.in_features
        else:
            # Different handling may be required for other types of layers
            target_input_dim = target_layer.in_features  # Implement according to specific layer type
        
        output_dim = jac.numel() // target_input_dim
        return jac.view(output_dim, target_input_dim)
    