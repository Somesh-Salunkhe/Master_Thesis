import numpy as np
import scipy.stats
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
import pandas as pd
from typing import Tuple, Optional

def apply_gmm(
    clu_feat: np.ndarray, 
    k: int, 
    cov_type: str = 'full', 
    seed: int = 1, 
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, GaussianMixture]:
    """
    Applies Gaussian Mixture Model clustering to the provided features.
    
    Args:
        clu_feat: Input features (N, D).
        k: Number of clusters.
        cov_type: Covariance type for GMM.
        seed: Random seed for reproducibility.
        **kwargs: Additional arguments for GaussianMixture.
        
    Returns:
        labels: Cluster assignments for each sample.
        distances: Distance of each sample to its cluster center.
        gmm: The fitted GMM model.
    """
    print(f"GMM clustering in progress (k={k})...")
    gmm = GaussianMixture(
        n_components=k, 
        covariance_type=cov_type, 
        random_state=seed, 
        **kwargs
    ).fit(clu_feat)
    
    # Computing distances to centroids
    # Centroids are determined by the sample with the highest density for each component
    centers = np.empty(shape=(gmm.n_components, clu_feat.shape[1]))
    for i in range(gmm.n_components):
        # Handle potential singular covariance
        try:
            density = scipy.stats.multivariate_normal(
                cov=gmm.covariances_[i], 
                mean=gmm.means_[i]
            ).logpdf(clu_feat)
            centers[i, :] = clu_feat[np.argmax(density)]
        except Exception:
            # Fallback to mean if covariance is singular
            centers[i, :] = gmm.means_[i]
            
    pair_dist = pairwise_distances(centers, clu_feat)

    # Assigning each embedding vector to a cluster
    labels = gmm.predict(clu_feat)

    # Computing the distances of each point to its respective cluster center
    distances = np.array([pair_dist[label, i] for i, label in enumerate(labels)])
    
    return labels, distances, gmm

def get_centroids(gmm: GaussianMixture, clu_feat: Optional[np.ndarray] = None) -> np.ndarray:
    """Returns the centroids (means) of the GMM components."""
    return gmm.means_

def create_correlation_matrix(
    activities: np.ndarray, 
    clusters: np.ndarray, 
    n_clusters: int, 
    n_classes: int, 
    distances: Optional[np.ndarray] = None, 
    normalize_by_distance: bool = False
) -> np.ndarray:
    """Creates a matrix representing the correlation between clusters and activity classes."""
    print("Creating Cluster-to-Activity Matrix...")
    
    # activities is expected to be one-hot or softmax probabilities (N, n_classes)
    # clusters is cluster IDs (N,)
    
    act_indices = np.argmax(activities, axis=1)
    dets = np.stack((act_indices, clusters), axis=1)
    
    corr_matrix = np.zeros((n_clusters, n_classes))
    
    for i in range(len(clusters)):
        c = int(clusters[i])
        a = int(act_indices[i])
        
        weight = 1.0
        if normalize_by_distance and distances is not None:
            weight = distances[i]
            
        corr_matrix[c, a] += weight
    
    return normalize(corr_matrix, axis=1, norm='l1')
