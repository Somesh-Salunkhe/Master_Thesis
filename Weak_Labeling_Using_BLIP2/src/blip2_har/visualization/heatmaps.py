import numpy as np
import plotly.graph_objects as go
from typing import List, Optional

def plot_heatmap(
    matrix: np.ndarray, 
    x_labels: List[str], 
    y_labels: List[str], 
    title: str = "Heatmap",
    colorscale: str = 'Viridis'
) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=matrix, x=x_labels, y=y_labels, colorscale=colorscale
    ))
    fig.update_layout(title=title, xaxis_tickangle=-45)
    return fig

def plot_probability_heatmap(probs, class_names, title="Probability Heatmap"):
    best_indices = probs.argmax(axis=1)
    y_labels = [f"C{i}: {class_names[best_indices[i]]}" for i in range(len(probs))]
    return plot_heatmap(probs, class_names, y_labels, title, colorscale='Plasma')
