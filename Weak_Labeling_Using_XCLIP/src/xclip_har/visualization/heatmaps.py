import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Optional

def plot_cosine_heatmap(
    sim_matrix: np.ndarray, 
    x_labels: List[str], 
    y_labels: Optional[List[str]] = None,
    title: str = "Cosine Similarity Heatmap"
) -> go.Figure:
    """Creates a Plotly heatmap for cosine similarity matrix."""
    
    if y_labels is None:
        y_labels = [f"C{i}" for i in range(sim_matrix.shape[0])]
        
    fig = px.imshow(
        sim_matrix,
        x=x_labels,
        y=y_labels,
        color_continuous_scale="Viridis",
        aspect="auto",
        title=title,
        labels=dict(x="Predicted Activity", y="Cluster ID", color="Cosine Similarity")
    )

    k = sim_matrix.shape[0]
    height = max(800, 15 * k)

    fig.update_layout(
        width=1200, 
        height=height, 
        xaxis_tickangle=-45
    )
    
    return fig
