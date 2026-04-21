# Master Thesis — Weak Annotation of Human Activity Datasets Using Vision Language Models

> **Author:** Somesh Salunkhe  
> **Institution:** University of Siegen  
> **Thesis Report:** [`Master_Thesis_Report.pdf`](./Master_Thesis_Report.pdf)

---

## 📌 Overview

This repository contains the full codebase, visualizations, and thesis report for a Master's thesis on Weak Annotation of Human Activity Datasets Using Vision Language Models. The project explores the use of large Vision-Language Models (VLMs) — specifically **BLIP-2** and **X-CLIP** — to generate weak annotations for activity classification **without requiring manual frame-level labels**.

The pipeline covers:
- Visual feature extraction from egocentric video using VLMs
- Sliding-window aggregation and GMM-based clustering
- Semantic label mapping via Image-Text Contrastive (ITC) alignment
- Embedding visualization using t-SNE / UMAP

---

## 🗂️ Repository Structure

```text
Master_Thesis/
├── Master_Thesis_Report.pdf                        # Full thesis document
├── Weak_Labeling_Using_BLIP2/                      # BLIP-2 based weak annotation pipeline
│   ├── src/blip2_har/                              # Core pipeline modules
│   │   ├── features/                               # Feature extraction & pooling
│   │   ├── clustering/                             # GMM clustering
│   │   ├── classification/                         # ITC text-to-cluster mapping
│   │   ├── captions/                               # Staged caption generation
│   │   └── utils/                                  # I/O, path, SRT helpers
│   ├── scripts/                                    # Entry point scripts
│   ├── configs/                                    # Pipeline configuration files
│   ├── requirements.txt
│   └── README.md
├── Weak_Labeling_Using_XCLIP/                      # X-CLIP based weak annotation pipeline
└── Weak_Annotation_Visualization (Author's Embeddings)/   # t-SNE/UMAP visualizations
```

---

## 🔬 Methodology

The thesis proposes an end-to-end **weak annotation framework** that eliminates the need for manual labels in egocentric activity recognition:

1. **Feature Extraction** — Extract frame-level or clip-level visual embeddings using BLIP-2 / X-CLIP
2. **Sliding Window Pooling** — Aggregate temporal frames using configurable window sizes and strides
3. **GMM Clustering** — Group unlabeled window embeddings into semantically coherent clusters
4. **ITC Classification** — Map clusters to activity labels using prompt-based text-image alignment
5. **Caption Generation** — Produce descriptive natural language captions for each cluster
6. **Visualization** — Render SRT subtitle overlays and interactive embedding plots (t-SNE/UMAP)

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- [PyTorch](https://pytorch.org/get-started/locally/)

### Installation

Navigate into either pipeline directory and install:

```bash
cd Weak_Labeling_Using_BLIP2
pip install -e .
```

Or for X-CLIP:

```bash
cd Weak_Labeling_Using_XCLIP
pip install -r requirements.txt
```

---

## 🛠️ BLIP-2 Pipeline Usage

```bash
# Step 1: Extract visual features
python scripts/run_extraction.py --dataset wear --gpu 0

# Step 2: Pool frames into windows
python scripts/run_pooling.py --dataset wear --window_s 4 --stride_s 1

# Step 3: Cluster window embeddings
python scripts/run_clustering.py --dataset wear --n_clusters 100

# Step 4: Map clusters to activity labels (ITC)
python scripts/run_classification.py --dataset wear --gpu 0

# Step 5: Generate captions
python scripts/run_captions.py --dataset wear --gpu 0

# Step 6: Visualize results (SRT or interactive plot)
python scripts/run_visualization.py --type srt --dataset wear --subject_id sbj_1
```

---

## 📊 Embedding Visualizations

The `Weak_Annotation_Visualization (Author's Embeddings)/` directory contains pre-computed t-SNE and UMAP plots of visual embeddings, allowing exploration of cluster separability and activity structure in the latent space.

---

## 🧠 Models Used

| Model | Role |
|-------|------|
| [BLIP-2](https://huggingface.co/Salesforce/blip2-opt-2.7b) | Feature extraction + ITC classification + captioning |
| [X-CLIP](https://huggingface.co/microsoft/xclip-base-patch32) | Video-text alignment for activity recognition |
| GMM (Gaussian Mixture Model) | Unsupervised temporal clustering |

---

## 📄 Thesis Report

The complete thesis report is available in this repository:  
📥 [`Master_Thesis_Report.pdf`](./Master_Thesis_Report.pdf)

---

## 📝 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

## 🙏 Acknowledgements

Special thanks to the supervisors and colleagues at the **University of Siegen** for their guidance throughout this research.
