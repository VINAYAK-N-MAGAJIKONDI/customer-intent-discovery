"""
Comprehensive Evaluation Suite for Unsupervised Customer Intent Discovery.

Calculates:
- Silhouette Score (on clustered queries, excluding noise)
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Semantic Coherence
- Noise & Outlier statistics
- Cluster Size Distribution & Balance Index
- Human-interpretable metric diagnostics
"""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def evaluate_intent_clustering(
    embeddings: np.ndarray,
    labels: np.ndarray,
    coherence_by_cluster: Optional[Dict[int, float]] = None
) -> Dict[str, Any]:
    """
    Compute rigorous evaluation metrics for discovered intent clusters.
    Carefully handles edge cases where metrics are mathematically undefined.
    """
    total_samples = len(labels)
    if total_samples == 0:
        return {"error": "Empty dataset provided."}

    noise_mask = (labels == -1)
    noise_count = int(np.sum(noise_mask))
    noise_pct = round((noise_count / total_samples) * 100, 2)

    valid_mask = ~noise_mask
    valid_samples = int(np.sum(valid_mask))
    valid_labels = labels[valid_mask]
    valid_embeddings = embeddings[valid_mask]

    unique_valid_clusters = sorted(list(set(valid_labels)))
    k_valid = len(unique_valid_clusters)

    metrics: Dict[str, Any] = {
        "total_samples": total_samples,
        "n_clusters": k_valid,
        "noise_count": noise_count,
        "noise_percentage": noise_pct,
        "clustered_samples": valid_samples,
        "clustered_percentage": round(100.0 - noise_pct, 2)
    }

    # Cluster size statistics
    if k_valid > 0:
        cluster_sizes = [int(np.sum(valid_labels == c)) for c in unique_valid_clusters]
        metrics["size_stats"] = {
            "mean": float(round(np.mean(cluster_sizes), 1)),
            "median": float(round(np.median(cluster_sizes), 1)),
            "min": int(np.min(cluster_sizes)),
            "max": int(np.max(cluster_sizes)),
            "std": float(round(np.std(cluster_sizes), 1)),
            "sizes": cluster_sizes
        }
    else:
        metrics["size_stats"] = {
            "mean": 0, "median": 0, "min": 0, "max": 0, "std": 0, "sizes": []
        }

    # Mean Intra-Cluster Semantic Coherence
    if coherence_by_cluster and len(coherence_by_cluster) > 0:
        metrics["mean_semantic_coherence"] = float(round(np.mean(list(coherence_by_cluster.values())), 4))
    else:
        metrics["mean_semantic_coherence"] = None

    # Mathematical checks for distance-based metrics (Requires at least 2 clusters and samples > K)
    can_eval_distance = (k_valid >= 2) and (valid_samples > k_valid)

    if can_eval_distance:
        try:
            sil = float(silhouette_score(valid_embeddings, valid_labels, metric="cosine"))
            metrics["silhouette_score"] = round(sil, 4)
            metrics["silhouette_status"] = "Valid"
        except Exception as e:
            metrics["silhouette_score"] = None
            metrics["silhouette_status"] = f"Calculation failed: {e}"

        try:
            dbi = float(davies_bouldin_score(valid_embeddings, valid_labels))
            metrics["davies_bouldin_index"] = round(dbi, 4)
            metrics["davies_bouldin_status"] = "Valid"
        except Exception as e:
            metrics["davies_bouldin_index"] = None
            metrics["davies_bouldin_status"] = f"Calculation failed: {e}"

        try:
            chi = float(calinski_harabasz_score(valid_embeddings, valid_labels))
            metrics["calinski_harabasz_index"] = round(chi, 2)
            metrics["calinski_harabasz_status"] = "Valid"
        except Exception as e:
            metrics["calinski_harabasz_index"] = None
            metrics["calinski_harabasz_status"] = f"Calculation failed: {e}"
    else:
        reason = (
            f"Requires at least 2 non-noise clusters (found {k_valid}) "
            f"with at least {k_valid + 1} queries."
        )
        metrics["silhouette_score"] = None
        metrics["silhouette_status"] = reason
        metrics["davies_bouldin_index"] = None
        metrics["davies_bouldin_status"] = reason
        metrics["calinski_harabasz_index"] = None
        metrics["calinski_harabasz_status"] = reason

    # Interpretability & quality assessment summary
    interpretations = {}
    if metrics.get("silhouette_score") is not None:
        s = metrics["silhouette_score"]
        if s > 0.45:
            interpretations["silhouette"] = "Excellent semantic separation between discovered intent groups."
        elif s > 0.25:
            interpretations["silhouette"] = "Good intent boundaries; mild semantic overlap on nuanced queries."
        elif s > 0.05:
            interpretations["silhouette"] = "Fair clustering; consider reviewing overlapping intents."
        else:
            interpretations["silhouette"] = "Weak separation; customer queries may be highly diffuse or homogeneous."

    if metrics.get("davies_bouldin_index") is not None:
        d = metrics["davies_bouldin_index"]
        if d < 1.2:
            interpretations["davies_bouldin"] = "Low inter-cluster similarity; intents are well-distinguished (Lower is better)."
        elif d < 2.0:
            interpretations["davies_bouldin"] = "Moderate inter-cluster similarity; typical for conversational domains."
        else:
            interpretations["davies_bouldin"] = "Elevated similarity; multiple intents share similar terminology."

    if noise_pct < 10.0:
        interpretations["noise"] = f"Low noise ({noise_pct}%); majority of queries cleanly mapped to intents."
    elif noise_pct < 30.0:
        interpretations["noise"] = f"Moderate noise ({noise_pct}%); common for exploratory real-world datasets."
    else:
        interpretations["noise"] = f"High noise ({noise_pct}%); consider tuning HDBSCAN 'Min Cluster Size' down."

    metrics["interpretations"] = interpretations
    return metrics
