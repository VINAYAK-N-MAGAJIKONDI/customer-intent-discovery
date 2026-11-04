"""
Automatic Intent Naming, Interpretation, and Semantic Overlap Engine.

Implements:
- c-TF-IDF (Class-based Term Frequency - Inverse Document Frequency)
- Centroid Medoid representative query extraction
- Human-interpretable intent name & description synthesis
- Cluster semantic coherence calculation
- Inter-cluster similarity matrix & overlap detection
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


def compute_ctfidf(
    clustered_docs: Dict[int, List[str]],
    top_n: int = 10,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 1
) -> Dict[int, List[Tuple[str, float]]]:
    """
    Calculate c-TF-IDF (Class-based TF-IDF) keyword weights for each intent cluster.
    
    W_{t, c} = TF_{t, c} * log(1 + A / TF_t)
    where A is the average number of words across all clusters.
    """
    cluster_ids = [c for c in sorted(clustered_docs.keys()) if c != -1]
    if not cluster_ids:
        return {}

    # Aggregate documents per cluster
    aggregated_texts = [" ".join(clustered_docs[c]) for c in cluster_ids]

    # Standard English stop words
    stop_words = "english"
    
    vec = CountVectorizer(
        ngram_range=ngram_range,
        stop_words=stop_words,
        min_df=min_df,
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b"
    )
    
    try:
        X = vec.fit_transform(aggregated_texts).toarray() # Shape: (n_clusters, n_vocab)
    except ValueError:
        # Fallback if vocabulary is too sparse or empty after stop words
        vec = CountVectorizer(ngram_range=(1, 1), min_df=1)
        X = vec.fit_transform(aggregated_texts).toarray()

    words = np.array(vec.get_feature_names_out())
    if len(words) == 0:
        return {c: [] for c in cluster_ids}

    # TF in each cluster
    tf = X
    # Total TF across all clusters for each term
    total_tf = np.sum(tf, axis=0) # Shape: (n_vocab,)
    
    # Average words per cluster
    words_per_cluster = np.sum(tf, axis=1) # Shape: (n_clusters,)
    A = np.mean(words_per_cluster) if len(words_per_cluster) > 0 else 1.0
    
    # Calculate c-TF-IDF matrix
    # Avoid div by zero
    idf = np.log(1.0 + (A / (total_tf + 1e-6)))
    ctfidf_matrix = tf * idf # Shape: (n_clusters, n_vocab)

    results: Dict[int, List[Tuple[str, float]]] = {}
    for idx, c in enumerate(cluster_ids):
        scores = ctfidf_matrix[idx]
        top_indices = np.argsort(scores)[::-1][:top_n]
        top_pairs = []
        for i in top_indices:
            if scores[i] > 0:
                top_pairs.append((str(words[i]), float(round(scores[i], 4))))
        results[c] = top_pairs

    return results


def synthesize_intent_label(
    top_keywords: List[Tuple[str, float]],
    medoid_query: str,
    cluster_id: int
) -> Tuple[str, str]:
    """
    Generate clean, business-friendly intent title and description.
    """
    if not top_keywords:
        # Fallback to medoid query snippet
        clean_med = re.sub(r"[^\w\s]", "", medoid_query).strip()
        words = clean_med.split()
        short_title = " ".join(words[:4]).title() if words else f"Intent Cluster {cluster_id}"
        return short_title, f"Inquiries regarding: {medoid_query}"

    kw_list = [k[0] for k in top_keywords]
    
    # Clean up keywords and build an intuitive title
    clean_kws = [k.title() for k in kw_list[:3]]
    if len(clean_kws) >= 2:
        title = f"{clean_kws[0]} & {clean_kws[1]}"
    else:
        title = f"{clean_kws[0]} Inquiry"
        
    description = (
        f"Customer queries primarily focusing on {', '.join(kw_list[:4])}. "
        f"Exemplified by: \"{medoid_query}\""
    )
    return title, description


def interpret_discovered_clusters(
    queries: List[str],
    embeddings: np.ndarray,
    labels: np.ndarray,
    probabilities: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Full interpretation pipeline:
    - Extracts c-TF-IDF keywords per cluster
    - Identifies top 3 centroid medoids per cluster
    - Generates human-readable labels and descriptions
    - Computes cluster coherence and inter-cluster similarity
    """
    n_samples = len(queries)
    if probabilities is None:
        probabilities = np.ones(n_samples, dtype=float)

    unique_labels = sorted(list(set(labels)))
    non_noise_labels = [c for c in unique_labels if c != -1]
    
    # Group queries and embeddings by cluster
    clustered_docs: Dict[int, List[str]] = {c: [] for c in unique_labels}
    clustered_indices: Dict[int, List[int]] = {c: [] for c in unique_labels}
    
    for idx, (q, c) in enumerate(zip(queries, labels)):
        clustered_docs[c].append(q)
        clustered_indices[c].append(idx)

    # c-TF-IDF keyword extraction
    kw_by_cluster = compute_ctfidf(clustered_docs, top_n=8)

    # Calculate centroids and medoids
    cluster_profiles: Dict[int, Dict[str, Any]] = {}
    centroids: Dict[int, np.ndarray] = {}

    for c in non_noise_labels:
        c_idxs = clustered_indices[c]
        c_embs = embeddings[c_idxs]
        
        # Centroid (mean vector)
        centroid = np.mean(c_embs, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroids[c] = centroid

        # Cosine similarity to centroid for all points in cluster
        sims_to_centroid = cosine_similarity(c_embs, centroid.reshape(1, -1)).flatten()
        coherence = float(np.mean(sims_to_centroid)) if len(sims_to_centroid) > 0 else 0.0

        # Sort indices by proximity to centroid
        sorted_internal = np.argsort(sims_to_centroid)[::-1]
        top_medoid_idxs = [c_idxs[i] for i in sorted_internal[:5]]
        
        medoid_query = queries[top_medoid_idxs[0]] if top_medoid_idxs else ""
        top_representative_queries = [
            {"query": queries[idx], "similarity": float(round(sims_to_centroid[i], 4))}
            for i, idx in enumerate(top_medoid_idxs)
        ]

        # Keywords
        kws = kw_by_cluster.get(c, [])
        intent_title, intent_desc = synthesize_intent_label(kws, medoid_query, c)

        cluster_profiles[c] = {
            "cluster_id": int(c),
            "intent_name": intent_title,
            "description": intent_desc,
            "representative_query": medoid_query,
            "top_queries": top_representative_queries,
            "keywords": kws,
            "size": len(c_idxs),
            "percentage": float(round((len(c_idxs) / n_samples) * 100, 2)),
            "semantic_coherence": float(round(coherence, 4)),
            "centroid": centroid
        }

    # Handle noise cluster (-1) if present
    noise_count = int(np.sum(labels == -1))
    noise_profile = None
    if noise_count > 0:
        noise_idxs = clustered_indices.get(-1, [])
        sample_noise = [queries[i] for i in noise_idxs[:10]]
        noise_profile = {
            "cluster_id": -1,
            "intent_name": "Unassigned / Outliers (Noise)",
            "description": "Queries that do not fit into any distinct semantic cluster with high confidence.",
            "representative_query": sample_noise[0] if sample_noise else "",
            "sample_queries": sample_noise,
            "size": noise_count,
            "percentage": float(round((noise_count / n_samples) * 100, 2))
        }

    # Inter-cluster similarity matrix (overlap detection)
    overlap_pairs: List[Dict[str, Any]] = []
    similarity_matrix = {}
    
    if len(non_noise_labels) > 1:
        centroid_keys = sorted(centroids.keys())
        matrix_embs = np.array([centroids[k] for k in centroid_keys])
        sim_matrix = cosine_similarity(matrix_embs)
        
        for i, c1 in enumerate(centroid_keys):
            for j, c2 in enumerate(centroid_keys):
                if i < j:
                    sim_val = float(sim_matrix[i, j])
                    if sim_val >= 0.70:
                        overlap_pairs.append({
                            "cluster_1_id": c1,
                            "cluster_1_name": cluster_profiles[c1]["intent_name"],
                            "cluster_2_id": c2,
                            "cluster_2_name": cluster_profiles[c2]["intent_name"],
                            "cosine_similarity": float(round(sim_val, 4)),
                            "overlap_level": "High Overlap" if sim_val >= 0.85 else "Moderate Overlap"
                        })
                        
        overlap_pairs.sort(key=lambda x: x["cosine_similarity"], reverse=True)

    return {
        "cluster_profiles": cluster_profiles,
        "noise_profile": noise_profile,
        "overlap_pairs": overlap_pairs,
        "n_clusters": len(cluster_profiles),
        "total_queries": n_samples
    }
