# X-CLIP HAR Pipeline

A modularized pipeline for generating weak annotations via X-CLIP video-text alignment, clustering, and semantic mapping.

## 🚀 Getting Started

### 1. Installation

```bash
# Install in editable mode
pip install -e .
```

### 2. Project Structure

```text
weak-labeling-xclip/
├── src/xclip_har/          # Core package logic
│   ├── features/           # Video preprocessing & extraction
│   ├── clustering/         # GMM clustering
│   ├── classification/     # Cluster-Text matching
│   └── utils/              # Path & Data I/O helpers
├── scripts/                # Entry point scripts
├── configs/                # Pipeline configurations
├── data/                   # Raw videos and processed features
└── output/                 # Generated results and visualizations
```

---

## 🛠️ Usage Workflow

### Step 0: Preprocessing
Resize videos and set consistent FPS for optimal feature extraction.
```bash
python scripts/run_preprocessing.py --dataset wear --fps 12
```

### Step 1: Feature Extraction
Extract X-CLIP embeddings for each preprocessed video.
```bash
python scripts/run_extraction.py --dataset wear --window_s 4 --stride_s 1
```

### Step 2: Clustering
Apply Gaussian Mixture Models (GMM) to the extracted features.
```bash
python scripts/run_clustering.py --dataset wear --n_clusters 100
```

### Step 3: Semantic Mapping
Match clusters to activity labels based on text prompts.
```bash
python scripts/run_classification.py --dataset wear --gpu 0
```

### Step 4: Visualization
Generate interactive visualizations and reports.
```bash
python scripts/run_visualization.py --type timeline --dataset wear --subject_id sbj_1
```

---

## 📂 Data Organization
- `data/[dataset]/raw/videos/`: Input `.mp4` files.
- `data/[dataset]/processed/xclip_embeddings/`: Extracted `.npy` features.
- `configs/prompts/`: JSON files containing activity class prompts.

---

## 📝 License
MIT License
