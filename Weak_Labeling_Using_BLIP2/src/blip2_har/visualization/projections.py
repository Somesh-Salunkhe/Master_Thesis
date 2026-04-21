import numpy as np
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from typing import Tuple

def perform_tsne(features, centroids, perplexity=30, seed=42):
    combined = np.vstack([features, centroids])
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed, init='pca', learning_rate='auto')
    emb = tsne.fit_transform(combined)
    return emb[:len(features)], emb[len(features):]

def plot_combined_projection(pts2d, cen2d, cluster_ids, text_labels, title="t-SNE Projection"):
    import plotly.express as px
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=pts2d[:,0], y=pts2d[:,1], mode='markers', marker=dict(color=cluster_ids, colorscale='Viridis'), name='Clusters'))
    fig.add_trace(go.Scattergl(x=cen2d[:,0], y=cen2d[:,1], mode='markers+text', marker=dict(symbol='star', size=12, color='black'), text=[f"C{i}" for i in range(len(cen2d))], name='Centroids'))
    fig.update_layout(title=title, template='plotly_white', width=1200, height=800)
    return fig
