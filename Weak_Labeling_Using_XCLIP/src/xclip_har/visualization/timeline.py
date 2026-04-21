import numpy as np
import plotly.graph_objects as go
from typing import List, Tuple

def labels_to_intervals(labels: List[str]) -> List[Tuple[int, int, str]]:
    """Converts a sequence of labels into intervals (start, end, label)."""
    if not labels:
        return []
    
    intervals = []
    start = 0
    current_label = labels[0]
    
    for i in range(1, len(labels)):
        if labels[i] != current_label:
            intervals.append((start, i, current_label))
            start = i
            current_label = labels[i]
            
    intervals.append((start, len(labels), current_label))
    return intervals

def plot_activity_timeline(
    gt_labels: List[str], 
    pred_labels: List[str], 
    title: str = "Activity Timeline"
) -> go.Figure:
    """Creates a Plotly figure comparing ground truth and predictions."""
    
    fig = go.Figure()
    
    # Plot Ground Truth
    gt_intervals = labels_to_intervals(gt_labels)
    for start, end, label in gt_intervals:
        fig.add_trace(go.Bar(
            name="GT: " + label,
            x=[end - start],
            y=["Ground Truth"],
            base=[start],
            orientation='h',
            showlegend=False,
            marker=dict(line=dict(width=0))
        ))
        
    # Plot Predictions
    pred_intervals = labels_to_intervals(pred_labels)
    for start, end, label in pred_intervals:
        fig.add_trace(go.Bar(
            name="Pred: " + label,
            x=[end - start],
            y=["Predictions"],
            base=[start],
            orientation='h',
            showlegend=False,
            marker=dict(line=dict(width=0))
        ))
        
    fig.update_layout(
        title=title,
        barmode='stack',
        xaxis_title="Time (Windows)",
        yaxis=dict(autorange="reversed"),
        height=400,
        showlegend=False
    )
    
    return fig
