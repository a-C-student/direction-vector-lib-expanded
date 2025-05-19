from torch import nn
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def find_layer_by_name(model: nn.Module, layer_name: str) -> int:
    """Find the index of a fully connected layer by its name"""
    layers = [name for name, _ in model.named_children() if isinstance(_, nn.Linear)]
    try:
        return layers.index(layer_name)
    except ValueError:
        raise ValueError(f"Fully connected layer named {layer_name} not found")    

def plot_neuron_activation_boxplot(
        data_df: pd.DataFrame,       # Input dataset (row index: sample names, column: neuron features)
        label_df: pd.DataFrame,       # Label dataset (contains sample names and corresponding tissue labels, index consistent with data_df)
        save_dir: str,                # Path to save the plot
        figsize: tuple = (12, 10),    # Figure size (width, height)
        ylabel: str = 'Neuron activation value',  # Y-axis label
        fontsize: int = 12,           # Font size for labels
        dpi: int = 300                # Resolution for saving
    ):
    """
    Plot boxplots of neuron activation values grouped by tissue labels.
    
    :param data_df: Input dataset with sample names as row index and neuron features as columns.
    :param label_df: Label dataset containing 'sample_name' (row index) and 'tissue_name' columns, indexed consistently with data_df.
    :param save_dir: Path to save the plots (created automatically if not exists).
    :param figsize: Figure dimensions (width, height), default (12, 10).
    :param ylabel: Label text for the Y-axis, default 'Neuron activation value'.
    :param fontsize: Font size for axis labels, default 12.
    :param dpi: Resolution for saving the figure, default 300.
    
    :return: None (figures are saved directly to the specified path)
    
    Exception Handling:
    - Check if data_df and label_df are pandas DataFrames.
    - Verify index consistency between data_df and label_df.
    - Create save_dir if it does not exist.
    - Handle index mismatch errors during grouping.
    """
    # Input type check
    if not isinstance(data_df, pd.DataFrame) or not isinstance(label_df, pd.DataFrame):
        raise TypeError("data_df and label_df must be pandas DataFrames")
    
    if not data_df.index.equals(label_df.index):
        raise ValueError("Indices (sample names) of data_df and label_df must be consistent")
    
    # Merge datasets and labels (aligned by index)
    combined_df = data_df.join(label_df, how='inner')
    
    for col in data_df.columns:
        try:
            feature_data = combined_df[col]
            tissue_labels = combined_df['tissue_name']  # Assume the tissue label column in label_df is 'tissue_name'
            
            # Group by tissue
            grouped = feature_data.groupby(tissue_labels)
            
            # Data validity check: ensure at least one sample per group
            if grouped.ngroups == 0:
                print(f"Warning: No valid grouped data for feature {col}, skipping plot")
                continue
            
            # Initialize figure
            plt.figure(figsize=figsize)
            
            boxplot_data = []
            boxplot_labels = []
            
            for group_name, group_data in grouped:
                boxplot_data.append(group_data.values)  # Extract numerical data
                boxplot_labels.append(f"{group_name}(n={len(group_data)})")
            
            # Sort groups by median
            sorted_indices = sorted(range(len(boxplot_data)), key=lambda k: np.median(boxplot_data[k]))
            sorted_boxplot_data = [boxplot_data[i] for i in sorted_indices]
            sorted_boxplot_labels = [boxplot_labels[i] for i in sorted_indices]
            
            # Plot boxplot with robust styling
            plt.boxplot(
                sorted_boxplot_data,
                labels=sorted_boxplot_labels,
                patch_artist=True,
                boxprops=dict(facecolor='white', linewidth=1.5),
                medianprops=dict(color='black', linewidth=1.5),
                showcaps=False,
                flierprops=dict(marker='o', markersize=3, markeredgecolor='none'),
                whiskerprops=dict(linewidth=1.5)
            )
            
            # Styling settings
            plt.grid(False)
            plt.xlabel('', fontsize=fontsize)
            plt.ylabel(ylabel, fontsize=fontsize, fontweight='bold', labelpad=15)
            plt.xticks(rotation=90, ha='center', fontsize=fontsize-2)  # Tilt labels to prevent overlap
            
            # Axis spine width
            ax = plt.gca()
            for axis in ['bottom', 'left']:
                ax.spines[axis].set_linewidth(1.5)
            
            # Save figure
            save_path = os.path.join(save_dir, f"neuron_{col}_boxplot.pdf")
            plt.tight_layout()
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            plt.close()
            
        except KeyError as e:
            raise KeyError(f"Missing required column in label data: {e}. Ensure label_df contains 'tissue_name'.")
        except Exception as e:
            print(f"Plotting failed for feature {col}. Error: {str(e)}")
            plt.close()
            continue