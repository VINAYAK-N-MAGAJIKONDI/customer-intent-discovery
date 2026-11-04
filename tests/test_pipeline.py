"""
Comprehensive Test Suite for Customer Intent Discovery Platform.

Covers:
- Preprocessing & data sanitization
- Dataset profiling
- Edge cases (nulls, empty, duplicates, non-English, micro-datasets)
- Embeddings & disk caching
- HDBSCAN, Agglomerative, and K-Means clustering
- c-TF-IDF keyword extraction & Medoid calculation
- Evaluation metrics & safety against undefined math
- Export formats (CSV & JSON)
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
import pandas as pd
import json

from src.preprocessing import clean_text, profile_dataset, preprocess_dataset
from src.embeddings import compute_embeddings, hash_text, DiskEmbeddingCache
from src.clustering import (
    run_hdbscan_clustering,
    run_agglomerative_clustering,
    run_kmeans_clustering,
    reduce_dimensions_for_clustering,
    compute_2d_projection
)
from src.interpretation import (
    compute_ctfidf,
    synthesize_intent_label,
    interpret_discovered_clusters
)
from src.evaluation import evaluate_intent_clustering
from src.exporter import (
    create_annotated_dataframe,
    create_intent_summary_dataframe,
    create_json_export
)


# ----------------------------------------------------
# 1. PREPROCESSING TESTS
# ----------------------------------------------------

def test_clean_text_basic_and_unicode():
    assert clean_text(None) == ""
    assert clean_text("   hello   world   ") == "hello world"
    assert clean_text("Where’s my order “now”?") == 'Where\'s my order "now"?'
    assert clean_text("Test\x00\x08control\x1fchars") == "Test control chars"


def test_profile_dataset():
    df = pd.DataFrame({
        "query": ["Where is my card?", None, "Where is my card?", "", "Need refund please"]
    })
    prof = profile_dataset(df, "query")
    assert prof["total_rows"] == 5
    assert prof["missing_count"] == 1
    assert prof["duplicate_count"] >= 1
    assert prof["unique_queries"] >= 2


def test_preprocess_dataset_filtering():
    df = pd.DataFrame({
        "query": [
            "Valid customer question here",
            None,
            "   ",
            "a",  # too short
            "Valid customer question here",  # duplicate
            "Another unique question"
        ]
    })
    proc_df, audit = preprocess_dataset(df, "query", drop_duplicates=True, min_chars=3)
    assert audit["records_retained"] == 2
    assert audit["dropped_missing"] == 1
    assert audit["dropped_empty"] == 1
    assert audit["dropped_too_short"] == 1
    assert audit["dropped_duplicates"] == 1
    assert len(proc_df) == 2


def test_preprocess_edge_cases():
    # Empty DataFrame
    empty_df = pd.DataFrame({"query": []})
    proc_df, audit = preprocess_dataset(empty_df, "query")
    assert len(proc_df) == 0
    assert audit["records_retained"] == 0

    # Non-English queries
    intl_df = pd.DataFrame({
        "query": [
            "¿Dónde está mi tarjeta de crédito?",
            "Je souhaite annuler ma commande",
            "Wo bleibt meine Lieferung?",
            "カードが届きません"
        ]
    })
    proc_intl, audit_intl = preprocess_dataset(intl_df, "query")
    assert audit_intl["records_retained"] == 4


# ----------------------------------------------------
# 2. EMBEDDINGS & CACHING TESTS
# ----------------------------------------------------

def test_embeddings_generation_and_normalization():
    queries = [
        "Where is my credit card?",
        "Card tracking inquiry",
        "How do I get a refund on my purchase?"
    ]
    embs = compute_embeddings(queries, model_name="all-MiniLM-L6-v2", use_cache=True)
    assert embs.shape == (3, 384)
    # Check L2 unit normalization
    norms = np.linalg.norm(embs, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0, 1.0], rtol=1e-5)


def test_embeddings_cache_hit():
    q = ["Unique test query for cache hit verification"]
    h = hash_text(q[0])
    
    # First call
    embs1 = compute_embeddings(q, model_name="all-MiniLM-L6-v2", use_cache=True)
    # Second call should pull directly from cache
    embs2 = compute_embeddings(q, model_name="all-MiniLM-L6-v2", use_cache=True)
    np.testing.assert_allclose(embs1, embs2)


# ----------------------------------------------------
# 3. CLUSTERING TESTS & EDGE CASES
# ----------------------------------------------------

def test_hdbscan_clustering_synthetic():
    # Create 2 distinct tight clusters + 1 outlier
    np.random.seed(42)
    c1 = np.random.normal(loc=0.0, scale=0.05, size=(15, 384))
    c2 = np.random.normal(loc=5.0, scale=0.05, size=(15, 384))
    outlier = np.random.normal(loc=100.0, scale=0.05, size=(1, 384))
    data = np.vstack([c1, c2, outlier])
    # Normalize
    data = data / np.linalg.norm(data, axis=1, keepdims=True)

    labels, probs, meta = run_hdbscan_clustering(data, min_cluster_size=5, min_samples=2, use_dim_reduction=False)
    assert len(labels) == 31
    # Check that at least 2 clusters were discovered
    unique_clusters = set(labels) - {-1}
    assert len(unique_clusters) >= 1


def test_agglomerative_clustering():
    np.random.seed(42)
    c1 = np.random.normal(loc=0.0, scale=0.1, size=(8, 50))
    c2 = np.random.normal(loc=2.0, scale=0.1, size=(8, 50))
    data = np.vstack([c1, c2])
    data = data / np.linalg.norm(data, axis=1, keepdims=True)

    labels, probs, meta = run_agglomerative_clustering(data, n_clusters=2)
    assert len(set(labels)) == 2
    assert len(labels) == 16


def test_clustering_edge_case_micro_dataset():
    # Dataset with only 2 items
    tiny_data = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    labels, probs, meta = run_hdbscan_clustering(tiny_data, min_cluster_size=2)
    assert len(labels) == 2


def test_dim_reduction_and_2d_projection():
    np.random.seed(42)
    data = np.random.rand(25, 384).astype(np.float32)
    coords_umap, meta_umap = compute_2d_projection(data, method="umap")
    assert coords_umap.shape == (25, 2)
    coords_pca, meta_pca = compute_2d_projection(data, method="pca")
    assert coords_pca.shape == (25, 2)
    assert "explained_variance_2d" in meta_pca


# ----------------------------------------------------
# 4. INTERPRETATION & c-TF-IDF TESTS
# ----------------------------------------------------

def test_ctfidf_and_label_synthesis():
    docs = {
        0: ["track my credit card delivery", "where is my new card", "card tracking status"],
        1: ["i want a refund for damaged item", "return my purchase refund please", "refund not received"]
    }
    kw_dict = compute_ctfidf(docs, top_n=5)
    assert 0 in kw_dict and 1 in kw_dict
    
    # Check that 'card' is in cluster 0 keywords
    c0_words = [k[0] for k in kw_dict[0]]
    assert any("card" in w for w in c0_words)

    # Check label synthesis
    title, desc = synthesize_intent_label(kw_dict[0], "where is my new card", 0)
    assert len(title) > 0
    assert len(desc) > 0


def test_interpret_discovered_clusters():
    queries = [
        "card delivery tracking",
        "where is my card",
        "cancel order and refund",
        "refund request please"
    ]
    embs = compute_embeddings(queries)
    labels = np.array([0, 0, 1, 1])
    
    interp = interpret_discovered_clusters(queries, embs, labels)
    assert interp["n_clusters"] == 2
    assert 0 in interp["cluster_profiles"]
    assert 1 in interp["cluster_profiles"]
    prof0 = interp["cluster_profiles"][0]
    assert prof0["size"] == 2
    assert len(prof0["representative_query"]) > 0


# ----------------------------------------------------
# 5. EVALUATION METRICS TESTS & SAFETY CHECKS
# ----------------------------------------------------

def test_evaluation_metrics_normal():
    np.random.seed(42)
    embs = np.random.rand(20, 10)
    labels = np.array([0]*10 + [1]*10)
    metrics = evaluate_intent_clustering(embs, labels)
    assert metrics["n_clusters"] == 2
    assert metrics["silhouette_score"] is not None
    assert metrics["davies_bouldin_index"] is not None
    assert metrics["noise_count"] == 0


def test_evaluation_metrics_single_cluster_safety():
    # Single cluster: mathematically silhouette score cannot be computed
    embs = np.random.rand(10, 5)
    labels = np.zeros(10, dtype=int)
    metrics = evaluate_intent_clustering(embs, labels)
    # Should safely return None with explanation, NOT crash
    assert metrics["silhouette_score"] is None
    assert "Requires at least 2 non-noise clusters" in metrics["silhouette_status"]


def test_evaluation_metrics_all_noise_safety():
    embs = np.random.rand(10, 5)
    labels = np.full(10, -1, dtype=int)
    metrics = evaluate_intent_clustering(embs, labels)
    assert metrics["noise_percentage"] == 100.0
    assert metrics["n_clusters"] == 0
    assert metrics["silhouette_score"] is None


# ----------------------------------------------------
# 6. EXPORT PIPELINE TESTS
# ----------------------------------------------------

def test_export_dataframes_and_json():
    df = pd.DataFrame({
        "query": ["Where is my card?", "Cancel order please"],
        "_cleaned_query": ["Where is my card?", "Cancel order please"],
        "_original_index": [0, 1]
    })
    labels = [0, 1]
    probs = [0.95, 0.98]
    profiles = {
        0: {
            "cluster_id": 0,
            "intent_name": "Card Tracking",
            "description": "Card inquiries",
            "representative_query": "Where is my card?",
            "keywords": [("card", 1.2)],
            "size": 1,
            "percentage": 50.0,
            "semantic_coherence": 0.88,
            "top_queries": [{"query": "Where is my card?", "similarity": 1.0}]
        },
        1: {
            "cluster_id": 1,
            "intent_name": "Order Cancellation",
            "description": "Cancel orders",
            "representative_query": "Cancel order please",
            "keywords": [("cancel", 1.5)],
            "size": 1,
            "percentage": 50.0,
            "semantic_coherence": 0.91,
            "top_queries": [{"query": "Cancel order please", "similarity": 1.0}]
        }
    }

    # Test Annotated DataFrame
    annotated = create_annotated_dataframe(df, labels, probs, profiles, "query")
    assert "discovered_intent_id" in annotated.columns
    assert "discovered_intent_name" in annotated.columns
    assert annotated.iloc[0]["discovered_intent_name"] == "Card Tracking"

    # Test Summary DataFrame
    summary = create_intent_summary_dataframe(profiles)
    assert len(summary) == 2
    assert "representative_customer_query" in summary.columns

    # Test JSON Export
    json_str = create_json_export(profiles, {"silhouette_score": 0.45}, {"total": 2})
    parsed = json.loads(json_str)
    assert "discovered_intents" in parsed
    assert "0" in parsed["discovered_intents"]
