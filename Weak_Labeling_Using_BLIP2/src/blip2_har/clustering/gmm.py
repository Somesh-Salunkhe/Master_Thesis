import numpy as np
import scipy.stats
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances
from typing import Tuple, Optional

def apply_gmm(
    clu_feat: np.ndarray, 
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
    ).fit(clu_feat)
    
    centers = np.empty(shape=(gmm.n_components, clu_feat.shape[1]))
    for i in range(gmm.n_components):
        try:
            density = scipy.stats.multivariate_normal(
                cov=gmm.covariances_[i], 
                mean=gmm.means_[i]
            ).logpdf(clu_feat)
            centers[i, :] = clu_feat[np.argmax(density)]
        except Exception:
            centers[i, :] = gmm.means_[i]
            
    pair_dist = pairwise_distances(centers, clu_feat)
    labels = gmm.predict(clu_feat)
    distances = np.array([pair_dist[label, i] for i, label in enumerate(labels)])
    
    return labels, distances, gmm
