
# Direction Vector Analysis Library

A PyTorch-based library for analyzing directional vectors in neural networks, focusing on weight vector extraction, forward propagation, Jacobian matrix computation, and visualization of neuron activations.


## Table of Contents
- [Library Overview](#library-overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [1. Define a Neural Network](#1-define-a-neural-network)
  - [2. Initialize the Analyzer](#2-initialize-the-analyzer)
  - [3. Extract Weight Vectors](#3-extract-weight-vectors)
  - [4. Forward Propagation](#4-forward-propagation)
  - [5. Compute Jacobian Matrix](#5-compute-jacobian-matrix)
  - [6. Visualize Neuron Activations](#6-visualize-neuron-activations)
- [Core Components](#core-components)
  - [DirectionVectorAnalyzer Class](#directionvectoranalyzer-class)
  - [Utility Functions](#utility-functions)
- [API Reference](#api-reference)
- [Contribution](#contribution)
- [License](#license)


## Library Overview
This library provides tools to analyze the directional behavior of neurons in neural networks. It enables users to:
- Extract weight vectors from specific neurons in fully connected layers.
- Propagate these vectors through the network to observe their impact on output layers.
- Compute Jacobian matrices to understand input-output relationships at the neuron level.
- Visualize neuron activations across different tissue/organization groups using boxplots.

Designed for researchers and developers working on neural network interpretability, especially in bioinformatics and computational biology contexts where tissue-specific neuron analysis is critical.


## Key Features
1. **Weight Vector Extraction**: Retrieve weights of individual or multiple neurons in specified layers.
2. **Forward Propagation**: Simulate the effect of weight vectors through subsequent layers.
3. **Jacobian Computation**: Measure how output neurons change with respect to input neurons at a sample point.
4. **Activation Visualization**: Generate publication-ready boxplots of neuron activations grouped by tissue labels.
5. **Model Agnostic**: Works with any PyTorch model containing linear layers, activation functions, and batch normalization.

<!-- 
## Installation
### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- pandas, matplotlib, numpy

### Install via pip (coming soon)
```bash
pip install direction-vector-lib
```

### Manual Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/direction-vector-lib.git
   ```
2. Import modules in your project:
   ```python
   from direction_vector_lib.core import DirectionVectorAnalyzer
   from direction_vector_lib.utils import plot_neuron_activation_boxplot
   ``` -->


## Quick Start

### 1. Define a Neural Network
```python
import torch
from torch import nn

class DemoNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LeakyReLU(),
                nn.BatchNorm1d(dim)
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
```


### 2. Initialize the Analyzer
```python
model = DemoNetwork(input_dim=100, hidden_dims=[64, 32], output_dim=20).net
analyzer = DirectionVectorAnalyzer(model)
```


### 3. Extract Weight Vectors
```python
# Single neuron (index 5 in layer 3)
single_weight = analyzer.get_weight_vector(layer_idx=3, neuron_idx=5)
print(f"Single neuron weight shape: {single_weight.shape}")  # Output: torch.Size([32])

# Multiple neurons (indices 1-5 in layer 3)
multi_weight = analyzer.get_weight_vector(layer_idx=3, neuron_idx=[1, 2, 3, 4, 5])
print(f"Multiple neurons weight shape: {multi_weight.shape}")  # Output: torch.Size([5, 32])
```


### 4. Forward Propagation
```python
# Propagate from layer 4 to the output
output = analyzer.forward_propagate(
    weight_vector=multi_weight,
    start_layer_idx=4,
    elongation_ratio=10000  # Scale weight vector for numerical stability
)
print(f"Output layer shape: {output.shape}")  # Output: (1, 20)
```


### 5. Compute Jacobian Matrix
```python
import torch

# Generate sample input (batch size=1, input_dim=100)
sample_data = torch.randn(1, 100)

# Define target and output layers
target_layer = model.net[3]  # Linear layer (64→32)
output_layer = model.net[6]  # Final linear layer (32→20)

# Compute Jacobian matrix
jacobian = analyzer.compute_jacobian_at_sample(
    sample_data=sample_data,
    target_layer=target_layer,
    output_layer=output_layer
)
print(f"Jacobian shape: {jacobian.shape}")  # Output: torch.Size([20, 64])
```


### 6. Visualize Neuron Activations
```python
import pandas as pd

# Sample activation data (row: sample name, column: neuron features)
activation_df = pd.DataFrame(
    torch.randn(100, 5),
    index=[f"Sample_{i}" for i in range(100)],
    columns=[f"Neuron_{i}" for i in range(5)]
)

# Sample tissue labels (index must match activation_df)
label_df = pd.DataFrame(
    {"tissue_name": np.random.choice(["TissueA", "TissueB"], 100)},
    index=[f"Sample_{i}" for i in range(100)]
)

# Generate boxplots
plot_neuron_activation_boxplot(
    data_df=activation_df,
    label_df=label_df,
    save_dir="plots/",
    ylabel="Neuron Activation Value"
)
```


## Core Components

### DirectionVectorAnalyzer Class
#### Constructor
```python
def __init__(self, model: nn.Module):
    """
    Initialize the analyzer with a PyTorch model.
    :param model: Input neural network (contains linear, activation, and BN layers)
    """
```

#### Key Methods
1. **`get_weight_vector(layer_idx: int, neuron_idx: int/list)`**  
   Extracts weight vectors from specified neurons in a linear layer.  
   - `layer_idx`: Index of the target linear layer (0-based).  
   - `neuron_idx`: Single index (int) or list of indices for multiple neurons.  
   - Returns: `torch.Tensor` with shape `[N, next_layer_neurons]` (N=1 for single neuron).

2. **`forward_propagate(weight_vector: torch.Tensor, start_layer_idx: int)`**  
   Propagates weight vectors through subsequent layers.  
   - `weight_vector`: Input vector(s) from `get_weight_vector`.  
   - `start_layer_idx`: Layer index to begin propagation (e.g., `start_layer_idx=4` skips first 4 layers).  
   - Returns: Output activations as a `pandas.DataFrame`.

3. **`compute_jacobian_at_sample(sample_data: torch.Tensor, target_layer: nn.Module, output_layer: nn.Module=None)`**  
   Computes the Jacobian matrix at a specific sample point.  
   - `sample_data`: Input tensor with batch dimension `[1, input_dim]`.  
   - `target_layer`: Layer for which to compute input neuron gradients.  
   - `output_layer`: Target output layer (default: model's final output).  
   - Returns: Jacobian matrix `[output_dim, target_input_dim]`.


### Utility Functions
#### `find_layer_by_name(model: nn.Module, layer_name: str) -> int`  
Finds the index of a linear layer by its name (e.g., `"fc2"`).  
- Raises `ValueError` if the layer is not found.

#### `plot_neuron_activation_boxplot(data_df: pd.DataFrame, label_df: pd.DataFrame, save_dir: str)`  
Generates boxplots of neuron activations grouped by tissue labels.  
- Requires `data_df` and `label_df` to have matching indices and a `tissue_name` column in `label_df`.


## API Reference
| Module               | Description                                  |
|----------------------|----------------------------------------------|
| `DirectionVectorAnalyzer` | Core class for neural network analysis       |
| `find_layer_by_name`    | Helper to locate layers by name              |
| `plot_neuron_activation_boxplot` | Visualization of neuron activations          |

For detailed method signatures, see the [code documentation](https://github.com/your-username/direction-vector-lib/blob/main/direction_vector_lib/core.py).


<!-- ## Contribution
1. Fork the repository.
2. Create a new branch for your feature: `git checkout -b new-feature`
3. Commit changes with clear messages.
4. Submit a pull request for review.


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Contact
For bug reports or feature requests, open an issue on the [GitHub repository](https://github.com/your-username/direction-vector-lib/issues).

Developed by [山东大学], 2025. -->
