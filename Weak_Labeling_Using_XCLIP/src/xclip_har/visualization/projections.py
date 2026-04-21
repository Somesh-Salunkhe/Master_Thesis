import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from typing import List, Optional, Tuple

def perform_tsne(
    features: np.ndarray, 
    centroids: np.ndarray, 
    perplexity: int = 30, 
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Performs t-SNE on features and centroids."""
    Xn = normalize(features, axis=1)
    Cn = normalize(centroids, axis=1)
    combined = np.vstack([Xn, Cn])
    
    tsne = TSNE(
        n_components=2, 
        perplexity=perplexity, 
        init="pca",
        learning_rate="auto", 
        random_state=random_seed
    )
    
    emb = tsne.fit_transform(combined)
    pts2d = emb[:len(features)]
    cen2d = emb[len(features):]
    
    return pts2d, cen2d

def plot_combined_projection(
    pts2d: np.ndarray, 
    cen2d: np.ndarray, 
    labels: np.ndarray, 
    text_labels: np.ndarray, 
    title: str = "t-SNE Projection"
) -> go.Figure:
    """Creates a dual-plot figure for t-SNE projections."""
    
    palette = px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24
    vivid = px.colors.qualitative.Vivid + px.colors.qualitative.Bold

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Coloured by Cluster ID", "Coloured by Text Label"],
        horizontal_spacing=0.06
    )

    # Left: Clusters
    c_colors = [palette[int(cid) % len(palette)] for cid in labels]
    fig.add_trace(go.Scattergl(
        x=pts2d[:, 0], y=pts2d[:, 1],
        mode="markers",
        marker=dict(size=5, opacity=0.7, color=c_colors, line=dict(width=0.5, color="white")),
        name="Clusters",
        showlegend=False,
        hoverinfo="text",
        text=[f"Cluster {int(cid)}" for cid in labels],
    ), row=1, col=1)

    # Left: Centroids
    fig.add_trace(go.Scattergl(
        x=cen2d[:, 0], y=cen2d[:, 1],
        mode="markers+text",
        marker=dict(size=12, symbol="star", color="black", line=dict(width=1.5, color="white")),
        text=[f"C{i}" for i in range(len(cen2d))],
        textposition="top center", textfont=dict(size=9, color="black"),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1)

    # Right: Text Labels
    unique_labels = sorted(set(text_labels))
    for i, lbl in enumerate(unique_labels):
        mask = text_labels == lbl
        color = vivid[i % len(vivid)]
        fig.add_trace(go.Scattergl(
            x=pts2d[mask, 0], y=pts2d[mask, 1],
            mode="markers",
            marker=dict(size=5, opacity=0.7, color=color, line=dict(width=0.5, color="white")),
            name=lbl,
            legendgroup=lbl, showlegend=True,
            hoverinfo="text",
            text=[f"{lbl}<br>Cluster {int(cid)}" for cid in labels[mask]],
        ), row=1, col=2)

    # Right: Centroids
    fig.add_trace(go.Scattergl(
        x=cen2d[:, 0], y=cen2d[:, 1],
        mode="markers+text",
        marker=dict(size=12, symbol="star", color="black", line=dict(width=1.5, color="white")),
        text=[f"C{i}" for i in range(len(cen2d))],
        textposition="top center", textfont=dict(size=9, color="black"),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=2)

    fig.update_layout(
        title=title,
        template="plotly_white",
        width=2400, height=1000,
        legend=dict(
            title_text="Toggle traces:",
            font=dict(size=10), itemsizing="constant",
            itemclick="toggle", itemdoubleclick="toggleothers",
        ),
    )
    
    return fig
