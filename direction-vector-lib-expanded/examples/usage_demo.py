import torch
from torch import nn
from direction_vector_lib.core import DirectionVectorAnalyzer

# Define a simple multi-layer neural network (alternative to an autoencoder)
class DemoNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for i, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.LeakyReLU())
            layers.append(nn.BatchNorm1d(dim))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

# Initialize the model
model = DemoNetwork(input_dim=100, hidden_dims=[64, 32], output_dim=20).net
analyzer = DirectionVectorAnalyzer(model)

# Get the structure of each layer in the neural network
analyzer.get_network_structure()

# Extract the weight vector of the 5th neuron in the 2nd fully connected layer (index 1, assuming the first layer is the linear layer after the input layer)
weight_vector_single = analyzer.get_weight_vector(layer_idx=3, neuron_idx=5)
print(f"Direction vector: \n{weight_vector_single}")

weight_vector_list = analyzer.get_weight_vector(layer_idx=3, neuron_idx=[1,2,3,4,5])
# Propagate forward to the output layer
output = analyzer.forward_propagate(weight_vector_list, start_layer_idx=4)
print(f"Direction vector: \n{weight_vector_list}")
print(f"Output layer result of the direction vector: \n{output}")

# Compute the Jacobian matrix of the target layer's input neurons with respect to the output layer at a specified sample point
data = torch.randn(1, 100)
jacobian_matrix = analyzer.compute_jacobian_at_sample(sample_data=data, target_layer=model[3], output_layer=model[6])
print(f"Jacobian matrix: \n{jacobian_matrix}")