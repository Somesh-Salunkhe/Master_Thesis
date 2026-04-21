import numpy as np
import scipy.stats
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances
from typing import Tuple, Optional

def apply_gmm(
    features: np.ndarray, 
    k: int, 
    cov_type: str = 'full', 
    seed: int = 1, 
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, GaussianMixture]:
    """Applies GMM clustering to the provided features."""
    print(f"GMM clustering in progress (k={k})...")
    gmm = GaussianMixture(
        n_components=k, 
        covariance_type=cov_type, 
        random_state=seed, 
        **kwargs
    ).fit(features)
    
    # Calculate density centers (instead of just means)
    centers = np.empty(shape=(gmm.n_components, features.shape[1]))
    for i in range(gmm.n_components):
        try:
            density = scipy.stats.multivariate_normal(
                cov=gmm.covariances_[i], 
                mean=gmm.means_[i]
            ).logpdf(features)
            centers[i, :] = features[np.argmax(density)]
        except Exception:
            centers[i, :] = gmm.means_[i]
            
    pair_dist = pairwise_distances(centers, features)
    labels = gmm.predict(features)
    distances = np.array([pair_dist[label, i] for i, label in enumerate(labels)])
    
    return labels, distances, gmm
