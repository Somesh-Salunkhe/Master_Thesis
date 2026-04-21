import numpy as np
import plotly.graph_objects as go
from typing import List, Optional

def plot_heatmap(
    matrix: np.ndarray, 
    x_labels: List[str], 
    y_labels: List[str], 
    title: str = "Similarity Heatmap",
    colorscale: str = 'Viridis',
    z_title: str = 'Value'
) -> go.Figure:
    """Creates a Plotly heatmap from a matrix."""
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=x_labels,
        y=y_labels,
        colorscale=colorscale,
        colorbar=dict(title=z_title),
        hoverongaps=False,
    ))

    # Dynamic height based on number of rows
    num_rows = matrix.shape[0]
    height = max(800, 18 * num_rows)

    fig.update_layout(
        title=title,
        xaxis_title="Classes",
        yaxis_title="Clusters",
        xaxis={'side': 'bottom', 'tickangle': -45, 'tickfont': {'size': 10}},
        yaxis={'dtick': 1, 'tickfont': {'size': 9}},
        width=1200,
        height=height,
        margin=dict(l=250, r=50, b=150, t=100)
    )
    
    return fig

def plot_cosine_heatmap(
    cosine_sim: np.ndarray, 
    class_names: List[str], 
    cluster_names: Optional[List[str]] = None,
    title: str = "Cosine Similarity Heatmap"
) -> go.Figure:
    """Specialized heatmap for cosine similarity between clusters and classes."""
    
    if cluster_names is None:
        best_indices = cosine_sim.argmax(axis=1)
        cluster_names = [f"C{i}: {class_names[best_indices[i]]}" for i in range(len(cosine_sim))]
    
    return plot_heatmap(
        matrix=cosine_sim,
        x_labels=class_names,
        y_labels=cluster_names,
        title=title,
        z_title='Cosine Sim'
    )

def plot_probability_heatmap(
    probs: np.ndarray, 
    class_names: List[str], 
    cluster_names: Optional[List[str]] = None,
    title: str = "Softmax Probability Heatmap"
) -> go.Figure:
    """Specialized heatmap for softmax probabilities."""
    
    if cluster_names is None:
        best_indices = probs.argmax(axis=1)
        cluster_names = [f"C{i}: {class_names[best_indices[i]]}" for i in range(len(probs))]
        
    return plot_heatmap(
        matrix=probs,
        x_labels=class_names,
        y_labels=cluster_names,
        title=title,
        colorscale='Plasma',
        z_title='Probability'
    )
