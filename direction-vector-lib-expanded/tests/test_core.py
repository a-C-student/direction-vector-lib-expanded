import pytest
import torch
from torch import nn
from direction_vector_lib.core import DirectionVectorAnalyzer

def test_weight_vector_extraction():
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 3)
    )
    analyzer = DirectionVectorAnalyzer(model)
    # Extract the weight of the 2nd neuron in the first layer (index 0) (shape should be [5])
    weight = analyzer.get_weight_vector(layer_idx=0, neuron_idx=[1,2,3])
    assert weight.shape == torch.Size([3,5]), "Weight vector shape mismatch"

def test_forward_propagation():
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 3)
    )
    analyzer = DirectionVectorAnalyzer(model)
    weight = analyzer.get_weight_vector(layer_idx=0, neuron_idx=1)
    output = analyzer.forward_propagate(weight, start_layer_idx=1)
    assert output.shape == torch.Size([1, 3]), "Output dimension mismatch"

def compute_jacobian_at_sample():
    class GeneralNetwork(nn.Module):
        def __init__(self):
            super(GeneralNetwork, self).__init__()
            self.fc1 = nn.Linear(100, 64)
            self.relu1 = nn.ReLU()
            self.bn1 = nn.BatchNorm1d(64)
            self.fc2 = nn.Linear(64, 32)
            self.relu2 = nn.ReLU()
            self.bn2 = nn.BatchNorm1d(32)
            self.fc3 = nn.Linear(32, 10)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.bn1(x)
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.bn2(x)
            x = self.fc3(x)
            return x
        
    # Create and initialize the network
    model = GeneralNetwork()
    # Generate sample data (shape: [1, 100])
    data = torch.randn(1, 100)
    analyzer = DirectionVectorAnalyzer(model)
    # Compute the Jacobian matrix of the second linear layer (fc2) with respect to the output layer (fc3)
    target_layer = model.fc2
    output_layer = model.fc3
    jacobian_matrix = analyzer.compute_jacobian_at_sample(
        sample_data=data,
        target_layer=target_layer,
        output_layer=output_layer
    )
    assert jacobian_matrix.shape == torch.Size([10, 64]), "Jacobian matrix shape mismatch"

test_weight_vector_extraction()
test_forward_propagation()
compute_jacobian_at_sample()

