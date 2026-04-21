# BLIP-2 HAR Pipeline

A modularized pipeline for generating weak annotations via BLIP-2 feature extraction, clustering, and semantic mapping.

## 🚀 Getting Started

### 1. Installation

```bash
# Install in editable mode
pip install -e .
```

### 2. Project Structure

```text
weak-labeling-blip2/
├── src/blip2_har/          # Core package logic
│   ├── features/           # Feature extraction & pooling
│   ├── clustering/         # GMM clustering
│   ├── classification/     # ITC text mapping
│   ├── captions/           # Staged caption generation
│   └── utils/              # Path, Data I/O, and SRT helpers
├── scripts/                # Entry point scripts
├── configs/                # Pipeline configurations
├── data/                   # Raw videos and processed features
└── outputs/                # Generated results and visualizations
```

---

## 🛠️ Usage Workflow

### Step 1: Feature Extraction
Extract framewise visual embeddings from videos.
```bash
python scripts/run_extraction.py --dataset wear --gpu 0
```

### Step 2: Average Pooling
Aggregate frames into sliding windows (e.g., 4s windows with 1s stride).
```bash
python scripts/run_pooling.py --dataset wear --window_s 4 --stride_s 1
```

### Step 3: Clustering
Group window-wise features into clusters.
```bash
python scripts/run_clustering.py --dataset wear --n_clusters 100
```

### Step 4: ITC Classification
Map clusters to semantic labels using BLIP-2 ITC shared space.
```bash
python scripts/run_classification.py --dataset wear --gpu 0
```

### Step 5: Caption Generation
Generate descriptive captions for cluster samples.
```bash
python scripts/run_captions.py --dataset wear --gpu 0
```

### Step 6: Visualization
Generate SRT subtitles for video overlay or interactive plots.
```bash
python scripts/run_visualization.py --type srt --dataset wear --subject_id sbj_1
```

---

## 📂 Data Organization
- `data/[dataset]/raw/videos/`: Input `.mp4` files.
- `data/[dataset]/processed/blip2_features/`: Extracted `.npy` features.
- `configs/prompts/`: JSON files containing activity class prompts.

---

## 📝 License
MIT License
