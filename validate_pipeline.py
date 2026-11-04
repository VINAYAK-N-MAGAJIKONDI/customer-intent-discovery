"""
End-to-End Headless Pipeline Verification Script.

Executes the complete customer intent discovery workflow from data loading
to clustering, labeling, evaluation, and export. Used for CI/CD and smoke testing.
"""

import sys
import time
from pathlib import Path
import pandas as pd

# Add root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.preprocessing import preprocess_dataset, profile_dataset
from src.embeddings import compute_embeddings
from src.clustering import run_hdbscan_clustering, compute_2d_projection
from src.interpretation import interpret_discovered_clusters
from src.evaluation import evaluate_intent_clustering
from src.exporter import create_annotated_dataframe, create_intent_summary_dataframe


def run_pipeline_check():
    print("=" * 70)
    print("CUSTOMER INTENT DISCOVERY PLATFORM — E2E VERIFICATION CHECK")
    print("=" * 70)

    # 1. Load Data
    sample_path = ROOT_DIR / "data" / "banking77_sample.csv"
    if not sample_path.exists():
        sample_path = ROOT_DIR / "data" / "ecommerce_support_sample.csv"

    print(f"\n[1/6] Ingesting dataset from: {sample_path.name}")
    raw_df = pd.read_csv(sample_path)
    # Take 300 queries for rapid smoke testing
    test_df = raw_df.head(300).copy()
    print(f"      Total input records: {len(test_df)}")

    # 2. Preprocess & Profile
    print("\n[2/6] Running data sanitization and profiling...")
    prof = profile_dataset(test_df, "query")
    proc_df, audit = preprocess_dataset(test_df, "query", drop_duplicates=True, min_chars=3)
    print(f"      Retained: {len(proc_df)} records ({audit['retention_rate_pct']}%)")
    print(f"      Avg word count: {prof['avg_word_count']:.1f}")

    # 3. Embeddings
    print("\n[3/6] Computing semantic embeddings (all-MiniLM-L6-v2)...")
    t0 = time.time()
    texts = proc_df["_cleaned_query"].tolist()
    embeddings = compute_embeddings(texts, model_name="all-MiniLM-L6-v2", use_cache=True)
    print(f"      Generated embeddings shape: {embeddings.shape} in {time.time()-t0:.2f}s")

    # 4. Clustering (HDBSCAN)
    print("\n[4/6] Executing HDBSCAN clustering with UMAP manifold reduction...")
    t0 = time.time()
    labels, probs, meta = run_hdbscan_clustering(
        embeddings, min_cluster_size=6, min_samples=2, use_dim_reduction=True
    )
    n_clusters = meta["n_clusters_found"]
    noise_count = meta["noise_count"]
    print(f"      Discovered {n_clusters} clusters ({noise_count} noise queries) in {time.time()-t0:.2f}s")
    assert n_clusters >= 2, f"Expected at least 2 clusters, got {n_clusters}"

    # 5. Intent Interpretation & Labeling
    print("\n[5/6] Synthesizing intent names & extracting c-TF-IDF keywords...")
    interp = interpret_discovered_clusters(texts, embeddings, labels, probs)
    print(f"      Successfully named {len(interp['cluster_profiles'])} intent segments.")
    for c, p in list(interp["cluster_profiles"].items())[:5]:
        kw_str = ", ".join([k[0] for k in p["keywords"][:4]])
        print(f"      - Intent #{c}: '{p['intent_name']}' ({p['size']} queries, Coherence: {p['semantic_coherence']:.3f})")
        print(f"        Keywords: [{kw_str}]")
        print(f"        Medoid:   \"{p['representative_query'][:60]}...\"")

    # 6. Evaluation & 2D Projection
    print("\n[6/6] Computing 2D projections and evaluation metrics...")
    coords, _ = compute_2d_projection(embeddings, method="umap")
    coherence_map = {c: p["semantic_coherence"] for c, p in interp["cluster_profiles"].items()}
    eval_metrics = evaluate_intent_clustering(embeddings, labels, coherence_map)
    print(f"      Silhouette Score:     {eval_metrics.get('silhouette_score')}")
    print(f"      Davies-Bouldin Index: {eval_metrics.get('davies_bouldin_index')}")
    print(f"      Mean Coherence:       {eval_metrics.get('mean_semantic_coherence')}")

    # 7. Export test
    annotated = create_annotated_dataframe(proc_df, labels, probs, interp["cluster_profiles"], "query")
    summary = create_intent_summary_dataframe(interp["cluster_profiles"], interp.get("noise_profile"))
    assert len(annotated) == len(proc_df)
    assert len(summary) >= n_clusters

    print("\n" + "=" * 70)
    print("SUCCESS: End-to-end intent discovery pipeline verified and operational!")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline_check()
