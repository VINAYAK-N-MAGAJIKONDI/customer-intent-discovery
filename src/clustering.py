"""
Advanced Clustering and Manifold Learning Engine for Customer Intent Discovery.

Provides:
- UMAP manifold reduction for improved density-based clustering
- HDBSCAN for automatic intent discovery with noise isolation
- Adaptive fallback to Agglomerative Clustering for micro-datasets
- Optional K-Means baseline
- 2D Projections (UMAP & PCA) for interactive scatter visualizations
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings


def reduce_dimensions_for_clustering(
    embeddings: np.ndarray,
    n_components: int = 5,
    random_state: int = 42
) -> np.ndarray:
    """
    Project 384-d embeddings to a lower-dimensional manifold (5-10 dims)
    using UMAP to overcome the curse of dimensionality for density estimation.
    """
    n_samples = embeddings.shape[0]
    if n_samples <= n_components:
        # Cannot reduce if samples <= components
        return embeddings

    try:
        import umap
        n_neighbors = min(15, max(2, n_samples - 1))
        n_comp = min(n_components, n_samples - 2)
        if n_comp < 2:
            return embeddings

        reducer = umap.UMAP(
            n_components=n_comp,
            n_neighbors=n_neighbors,
            min_dist=0.0,
            metric="cosine",
            random_state=random_state,
            n_jobs=1
        )
        return reducer.fit_transform(embeddings)
    except Exception as e:
        warnings.warn(f"UMAP reduction fallback to PCA due to: {e}")
        pca_comp = min(n_components, n_samples - 1, embeddings.shape[1])
        return PCA(n_components=pca_comp, random_state=random_state).fit_transform(embeddings)


def compute_2d_projection(
    embeddings: np.ndarray,
    method: str = "umap",
    random_state: int = 42
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute 2D coordinates (x, y) for interactive visual cluster maps.
    
    Returns:
        coords_2d: np.ndarray of shape (N, 2)
        metadata: Dict containing method info and explained variance (for PCA)
    """
    n_samples = embeddings.shape[0]
    if n_samples < 2:
        return np.zeros((n_samples, 2), dtype=np.float32), {"method": "identity"}

    if method.lower() == "pca" or n_samples < 4:
        pca = PCA(n_components=2, random_state=random_state)
        coords = pca.fit_transform(embeddings)
        exp_var = float(np.sum(pca.explained_variance_ratio_))
        return coords, {"method": "PCA", "explained_variance_2d": exp_var}

    # Default: UMAP 2D
    try:
        import umap
        n_neighbors = min(15, max(2, n_samples - 1))
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric="cosine",
            random_state=random_state,
            n_jobs=1
        )
        coords = reducer.fit_transform(embeddings)
        return coords, {"method": "UMAP", "n_neighbors": n_neighbors}
    except Exception as e:
        warnings.warn(f"UMAP 2D failed ({e}); falling back to PCA.")
        pca = PCA(n_components=2, random_state=random_state)
        coords = pca.fit_transform(embeddings)
        exp_var = float(np.sum(pca.explained_variance_ratio_))
        return coords, {"method": "PCA (fallback)", "explained_variance_2d": exp_var}


def run_hdbscan_clustering(
    embeddings: np.ndarray,
    min_cluster_size: Optional[int] = None,
    min_samples: Optional[int] = None,
    cluster_selection_method: str = "eom",
    use_dim_reduction: bool = True,
    dim_components: int = 5,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Discover clusters using HDBSCAN with automatic parameter tuning.
    
    Returns:
        labels: 1D array of cluster IDs (-1 for noise)
        probabilities: 1D array of cluster assignment membership confidences
        meta: Execution metadata and parameter diagnostics
    """
    n_samples = embeddings.shape[0]
    if n_samples < 3:
        # Micro-dataset cannot form clusters
        return np.zeros(n_samples, dtype=int), np.ones(n_samples, dtype=float), {"algorithm": "trivial"}

    # Dynamic default for min_cluster_size if unspecified
    if min_cluster_size is None or min_cluster_size <= 1:
        if n_samples < 50:
            min_cluster_size = 3
        elif n_samples < 200:
            min_cluster_size = 5
        elif n_samples < 1000:
            min_cluster_size = 10
        else:
            min_cluster_size = 15

    min_cluster_size = max(2, min(min_cluster_size, n_samples // 2))
    
    if min_samples is None:
        min_samples = max(1, min_cluster_size // 2)

    clustering_features = embeddings
    if use_dim_reduction and n_samples >= 10:
        clustering_features = reduce_dimensions_for_clustering(
            embeddings, n_components=dim_components, random_state=random_state
        )

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
        metric="euclidean"
    )
    labels = clusterer.fit_predict(clustering_features)
    probabilities = getattr(clusterer, "probabilities_", np.ones(n_samples, dtype=float))

    unique_labels = set(labels)
    unique_non_noise = [c for c in unique_labels if c != -1]

    # Adaptive fallback: If HDBSCAN discovered 0 clusters or labeled > 80% as noise
    noise_ratio = np.mean(labels == -1) if len(labels) > 0 else 1.0
    fallback_triggered = False

    if len(unique_non_noise) < 2 and n_samples >= 6:
        fallback_triggered = True
        # Try smaller min_cluster_size
        fallback_min = max(2, min_cluster_size // 2)
        if fallback_min < min_cluster_size:
            clusterer2 = HDBSCAN(
                min_cluster_size=fallback_min,
                min_samples=1,
                cluster_selection_method=cluster_selection_method,
                metric="euclidean"
            )
            labels2 = clusterer2.fit_predict(clustering_features)
            probs2 = getattr(clusterer2, "probabilities_", probabilities)
            if len([c for c in set(labels2) if c != -1]) >= 2:
                labels = labels2
                probabilities = probs2
                min_cluster_size = fallback_min

    meta = {
        "algorithm": "HDBSCAN",
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "selection_method": cluster_selection_method,
        "dimension_reduced": use_dim_reduction and n_samples >= 10,
        "n_clusters_found": len([c for c in set(labels) if c != -1]),
        "noise_count": int(np.sum(labels == -1)),
        "noise_ratio": float(np.mean(labels == -1)),
        "fallback_triggered": fallback_triggered
    }

    return labels, probabilities, meta


def run_agglomerative_clustering(
    embeddings: np.ndarray,
    n_clusters: Optional[int] = None,
    distance_threshold: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Hierarchical Agglomerative Clustering using Cosine Distance.
    If n_clusters is None and distance_threshold is None, optimizes K via Silhouette sweep.
    """
    n_samples = embeddings.shape[0]
    if n_samples < 2:
        return np.zeros(n_samples, dtype=int), np.ones(n_samples, dtype=float), {"algorithm": "agglomerative"}

    # Automatic K selection if unspecified
    if n_clusters is None and distance_threshold is None:
        best_k = 2
        best_score = -1.0
        max_k = min(20, max(3, n_samples // 4))
        
        for k in range(2, max_k + 1):
            model = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
            temp_labels = model.fit_predict(embeddings)
            if len(set(temp_labels)) > 1:
                score = silhouette_score(embeddings, temp_labels, metric="cosine")
                if score > best_score:
                    best_score = score
                    best_k = k
        n_clusters = best_k

    if distance_threshold is not None:
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="cosine",
            linkage="average"
        )
    else:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="average"
        )

    labels = model.fit_predict(embeddings)
    probabilities = np.ones(n_samples, dtype=float)

    meta = {
        "algorithm": "Agglomerative",
        "n_clusters": len(set(labels)),
        "linkage": "average",
        "metric": "cosine"
    }
    return labels, probabilities, meta


def run_kmeans_clustering(
    embeddings: np.ndarray,
    n_clusters: int = 5,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    K-Means clustering baseline.
    """
    n_samples = embeddings.shape[0]
    k = min(n_clusters, max(1, n_samples))
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = model.fit_predict(embeddings)
    
    # Distance to centroid as confidence proxy
    distances = model.transform(embeddings)
    closest_dist = distances[np.arange(n_samples), labels]
    # Invert distance for confidence
    max_d = np.max(closest_dist) if np.max(closest_dist) > 0 else 1.0
    probabilities = 1.0 - (closest_dist / (max_d + 1e-6))
    
    meta = {
        "algorithm": "K-Means",
        "n_clusters": k,
        "inertia": float(model.inertia_)
    }
    return labels, probabilities, meta
