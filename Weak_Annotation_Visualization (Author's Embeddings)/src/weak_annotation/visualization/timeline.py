import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any, Optional

def labels_to_intervals(labels: List[str]) -> List[Dict[str, Any]]:
    """Converts a sequence of labels to start-end intervals."""
    if not labels: return []
    intervals = []
    start = 0
    curr_label = labels[0]
    for i, l in enumerate(labels):
        if l != curr_label:
            intervals.append({"Start": start, "End": i, "Activity": curr_label})
            start = i
            curr_label = l
    intervals.append({"Start": start, "End": len(labels), "Activity": curr_label})
    return intervals

def calculate_activity_accuracy(gt_list: List[str], pred_list: List[str]) -> float:
    """Calculates accuracy excluding background/null labels."""
    valid_gt = []
    valid_pred = []
    ignore_labels = ["background", "null/other", "neutral", "none", ""]
    
    for g, p in zip(gt_list, pred_list):
        if str(g).lower() in ignore_labels:
            continue
        valid_gt.append(str(g).strip().lower())
        valid_pred.append(str(p).strip().lower())
    
    if not valid_gt:
        return 0.0
    
    correct = sum([g == p for g, p in zip(valid_gt, valid_pred)])
    return correct / len(valid_gt)

def plot_activity_timeline(
    gt_labels: List[str], 
    pred_labels: List[str], 
    title: str = "Activity Timeline Comparison",
    accuracy: Optional[float] = None
) -> go.Figure:
    """Creates a Plotly figure comparing ground truth and predicted timelines."""
    
    gt_intervals = labels_to_intervals(gt_labels)
    pr_intervals = labels_to_intervals(pred_labels)
    
    for i in gt_intervals: i["Type"] = "Ground Truth"
    for i in pr_intervals: i["Type"] = "Predicted"
    
    plot_df = pd.DataFrame(gt_intervals + pr_intervals)
    plot_df["Duration"] = plot_df["End"] - plot_df["Start"]

    # Colors
    vivid = px.colors.qualitative.Vivid + px.colors.qualitative.Bold + px.colors.qualitative.Dark24
    unique_acts = sorted(plot_df["Activity"].unique())
    color_map = {act: vivid[i % len(vivid)] for i, act in enumerate(unique_acts)}
    color_map["Background"] = "#f0f0f0"
    
    fig = go.Figure()
    
    for act in unique_acts:
        mask = plot_df["Activity"] == act
        df_sub = plot_df[mask]
        
        fig.add_trace(go.Bar(
            name=act,
            x=df_sub["Type"],
            y=df_sub["Duration"],
            base=df_sub["Start"],
            marker_color=color_map.get(act, "#333"),
            hovertemplate=(
                "<b>%{fullData.name}</b><br>" +
                "Start: %{base}s<br>" +
                "Duration: %{y}s<br>" +
                "<extra></extra>"
            ),
            width=0.4
        ))

    if accuracy is None:
        accuracy = calculate_activity_accuracy(gt_labels, pred_labels)

    fig.add_annotation(
        text=f"Activity Accuracy (Weighted): {accuracy:.2%}",
        xref="paper", yref="paper",
        x=0.5, y=1.07, showarrow=False,
        font=dict(size=18, color="firebrick", family="Arial Black"),
        bgcolor="white", bordercolor="firebrick", borderwidth=2, borderpad=8
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=24), y=0.98),
        barmode='stack',
        template="plotly_white",
        height=1000, width=800,
        yaxis=dict(title="Time (Seconds)", autorange="reversed", gridcolor="#eee"),
        xaxis=dict(title="", tickfont=dict(size=14, color="black")),
        legend=dict(title="Activities", font=dict(size=11), groupclick="toggleitem"),
        margin=dict(t=150, b=50, l=100, r=50)
    )
    
    return fig
