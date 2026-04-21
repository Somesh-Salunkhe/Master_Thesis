# Weak Annotation Pipeline

A modularized pipeline for generating weak annotations via clustering and CLIP-based text classification, including visualization and evaluation tools.

## 🚀 Getting Started

### 1. Installation

It is recommended to use a virtual environment:

```bash
# Install the package in editable mode
pip install -e .
```

Alternatively, install dependencies directly:
```bash
pip install -r requirements.txt
```

### 2. Project Structure

```text
weak-annotation-pipeline/
├── src/weak_annotation/    # Core package logic
│   ├── clustering/         # GMM clustering
│   ├── classification/     # CLIP text mapping
│   ├── visualization/      # Plotly-based plotting
│   └── utils/              # Data I/O and shared helpers
├── scripts/                # Entry point scripts
├── configs/                # Pipeline configurations (.yaml)
├── data/                   # Input features and annotations
└── output/                 # Generated results and visualizations
```

---

## 🛠️ Usage Workflow

Follow these steps to process your data and generate results.

### Step 1: Clustering
Generate cluster centroids and labels for subjects based on feature embeddings.

```bash
python scripts/run_clustering.py --dataset wear --features clip --clusters 100
```

### Step 2: Text Classification
Map the generated clusters to semantic activity labels using CLIP text prototypes.

```bash
python scripts/run_classification.py --dataset wear --model clip_clusters
```

### Step 3: Visualization
Generate interactive visualizations for analysis.

```bash
# Generate all visualizations for a specific subject
python scripts/run_visualization.py --type all --dataset wear --subject_id sbj_1

# Generate only the activity timeline
python scripts/run_visualization.py --type timeline --dataset wear --subject_id sbj_1
```

Visualizations will be saved in `output/visualizations/[dataset]/[subject_id]/`.

---

## 📂 Data Organization

- `data/[dataset]/annotations/`: Ground-truth CSV files and JSON splits.
- `data/[dataset]/processed/`: Feature embeddings (e.g., CLIP, DINO) as `.npy` files.
- `configs/`: YAML files defining paths and parameters for each dataset.

---

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
