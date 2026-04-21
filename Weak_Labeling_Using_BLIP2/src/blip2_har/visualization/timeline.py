import numpy as np
import plotly.graph_objects as go
from typing import List, Tuple

def labels_to_intervals(labels: List[str]) -> List[Tuple[int, int, str]]:
    if not labels: return []
    intervals = []
    start = 0
    curr = labels[0]
    for i in range(1, len(labels)):
        if labels[i] != curr:
            intervals.append((start, i, curr))
            start = i
            curr = labels[i]
    intervals.append((start, len(labels), curr))
    return intervals

def plot_activity_timeline(gt_labels, pred_labels, title="Activity Timeline"):
    fig = go.Figure()
    for start, end, label in labels_to_intervals(gt_labels):
        fig.add_trace(go.Bar(name="GT: "+label, x=[end-start], y=["GT"], base=[start], orientation='h', showlegend=False))
    for start, end, label in labels_to_intervals(pred_labels):
        fig.add_trace(go.Bar(name="Pred: "+label, x=[end-start], y=["Pred"], base=[start], orientation='h', showlegend=False))
    fig.update_layout(title=title, barmode='stack', height=400)
    return fig
