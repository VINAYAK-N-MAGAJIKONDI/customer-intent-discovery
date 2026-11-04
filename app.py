"""
Production-Ready Customer Intent Discovery Platform.

An intelligent, unsupervised NLP platform that automatically discovers,
clusters, names, interprets, and visualizes customer intents from raw, unlabeled queries.
"""

import sys
import os
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import io

from src.preprocessing import profile_dataset, preprocess_dataset, clean_text
from src.embeddings import compute_embeddings, SUPPORTED_MODELS
from src.clustering import (
    run_hdbscan_clustering,
    run_agglomerative_clustering,
    run_kmeans_clustering,
    compute_2d_projection
)
from src.interpretation import interpret_discovered_clusters
from src.evaluation import evaluate_intent_clustering
from src.visualization import (
    plot_cluster_map_2d,
    plot_intent_size_distribution,
    plot_cluster_keywords,
    plot_intercluster_similarity_heatmap
)
from src.exporter import (
    create_annotated_dataframe,
    create_intent_summary_dataframe,
    create_json_export
)

# Set page config
st.set_page_config(
    page_title="Customer Intent Discovery Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    /* Global Styles */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 12px;
    }
    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 0.05em;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
        margin-top: 4px;
    }
    .badge-keyword {
        display: inline-block;
        background-color: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
        padding: 2px 8px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 500;
        margin-right: 4px;
        margin-bottom: 4px;
    }
    .card-intent {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #2563eb;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .medoid-quote {
        background: #f8fafc;
        border-left: 3px solid #94a3b8;
        padding: 8px 14px;
        border-radius: 4px;
        font-style: italic;
        color: #334155;
        font-size: 0.92rem;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "selected_query_col" not in st.session_state:
    st.session_state.selected_query_col = None
if "processed_df" not in st.session_state:
    st.session_state.processed_df = None
if "audit_report" not in st.session_state:
    st.session_state.audit_report = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "discovery_results" not in st.session_state:
    st.session_state.discovery_results = None
if "labels" not in st.session_state:
    st.session_state.labels = None
if "probs" not in st.session_state:
    st.session_state.probs = None
if "eval_metrics" not in st.session_state:
    st.session_state.eval_metrics = None
if "coords_2d_umap" not in st.session_state:
    st.session_state.coords_2d_umap = None
if "coords_2d_pca" not in st.session_state:
    st.session_state.coords_2d_pca = None


def reset_discovery_state():
    """Clear downstream analysis when data changes."""
    st.session_state.processed_df = None
    st.session_state.audit_report = None
    st.session_state.embeddings = None
    st.session_state.discovery_results = None
    st.session_state.labels = None
    st.session_state.probs = None
    st.session_state.eval_metrics = None
    st.session_state.coords_2d_umap = None
    st.session_state.coords_2d_pca = None


# Sidebar Navigation & Benchmark Datasets
with st.sidebar:
    st.title("🎯 Intent Discovery")
    st.caption("Unsupervised Customer Query Intelligence")
    st.divider()

    st.subheader("📁 Data Source")
    data_source_mode = st.radio(
        "Select Input Method:",
        ["Upload CSV File", "Load Benchmark Sample"],
        help="Upload your own customer queries or load a verified real-world benchmark dataset."
    )

    if data_source_mode == "Upload CSV File":
        uploaded_file = st.file_uploader(
            "Upload customer queries (CSV)",
            type=["csv"],
            help="Upload a CSV with customer messages, tickets, or chatbot queries."
        )
        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                if st.session_state.raw_df is None or not uploaded_df.equals(st.session_state.raw_df):
                    st.session_state.raw_df = uploaded_df
                    reset_discovery_state()
                    st.success(f"Loaded {len(uploaded_df):,} records!")
            except Exception as e:
                st.error(f"Error parsing CSV: {e}")
    else:
        benchmark_choice = st.selectbox(
            "Choose Benchmark Dataset:",
            [
                "Banking77 (Retail Banking Queries, PolyAI)",
                "CLINC150 (Multi-Domain + Real Noise Queries)",
                "E-Commerce Support (Orders & Refunds)"
            ]
        )
        if st.button("Load Selected Benchmark", use_container_width=True, type="primary"):
            data_dir = ROOT_DIR / "data"
            if "Banking77" in benchmark_choice:
                sample_path = data_dir / "banking77_sample.csv"
            elif "CLINC150" in benchmark_choice:
                sample_path = data_dir / "clinc150_sample.csv"
            else:
                sample_path = data_dir / "ecommerce_support_sample.csv"

            if sample_path.exists():
                st.session_state.raw_df = pd.read_csv(sample_path)
                reset_discovery_state()
                st.success(f"Loaded benchmark dataset ({len(st.session_state.raw_df):,} queries)!")
            else:
                st.error("Sample dataset file not found.")

    st.divider()
    st.markdown("### ⚙️ Quick Settings")
    selected_model = st.selectbox(
        "Embedding Model:",
        list(SUPPORTED_MODELS.keys()),
        index=0,
        help="all-MiniLM-L6-v2 offers the best balance of speed and clustering quality on CPU."
    )
    dim_reduction_toggle = st.checkbox("UMAP Manifold Reduction", value=True, help="Reduces 384-d vectors to 5D manifold before HDBSCAN to boost cluster separation.")
    
    st.divider()
    st.markdown(
        "<div style='font-size: 0.8rem; color: #64748b; text-align: center;'>"
        "Production-Ready Customer Intent Discovery Platform<br>"
        "Zero Paid API Dependency • 100% Open Source"
        "</div>",
        unsafe_allow_html=True
    )

# Main Application Layout
st.markdown("<div class='main-header'>Customer Intent Discovery Platform</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>"
    "Transform unstructured customer inquiries into structured, named, and actionable intent clusters using unsupervised semantic intelligence."
    "</div>",
    unsafe_allow_html=True
)

# Tabs
tab_data, tab_discovery, tab_intents, tab_viz, tab_eval, tab_export, tab_docs = st.tabs([
    "📁 1. Dataset & Audit",
    "⚙️ 2. Discovery Engine",
    "🎯 3. Discovered Intents",
    "🗺️ 4. Visualizations",
    "📈 5. Quality Evaluation",
    "💾 6. Export Center",
    "📖 7. Documentation"
])

# ----------------------------------------------------
# TAB 1: DATASET UPLOAD & AUDIT
# ----------------------------------------------------
with tab_data:
    st.subheader("Dataset Ingestion and Preprocessing Audit")
    
    if st.session_state.raw_df is None:
        st.info("👈 Please upload a CSV file or select a verified benchmark dataset from the left sidebar to get started.")
        
        # Display Feature Highlights
        st.markdown("### Platform Workflow")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            #### 1. Ingestion & Quality Control
            - Flexible CSV parsing
            - Automated deduplication & text normalization
            - Identification of invalid/empty customer queries
            - Preserves context for deep transformer embeddings
            """)
        with col2:
            st.markdown("""
            #### 2. Unsupervised Discovery
            - High-speed semantic embedding (MiniLM / BGE)
            - Density-based discovery with **HDBSCAN**
            - No need to guess the number of intents
            - Explicit outlier and noise isolation
            """)
        with col3:
            st.markdown("""
            #### 3. Automatic Interpretation
            - **c-TF-IDF** keyword extraction
            - Centroid **Medoid** representative query
            - Clear business intent titles & descriptions
            - Semantic overlap detection across clusters
            """)
    else:
        raw_df = st.session_state.raw_df
        cols = list(raw_df.columns)
        
        # Intelligent column guessing (query, text, message, inquiry, etc.)
        default_col_idx = 0
        for i, c in enumerate(cols):
            if c.lower() in ["query", "text", "customer_query", "message", "inquiry", "question", "body"]:
                default_col_idx = i
                break
                
        c1, c2 = st.columns([2, 1])
        with c1:
            query_col = st.selectbox(
                "Select column containing customer queries:",
                cols,
                index=default_col_idx,
                help="The column holding raw customer text."
            )
            st.session_state.selected_query_col = query_col
            
        with c2:
            drop_dups = st.checkbox("Drop Exact Duplicate Queries", value=True)
            min_length = st.slider("Minimum Character Length", min_value=1, max_value=20, value=3)

        # Profile Dataset
        profile = profile_dataset(raw_df, query_col)
        
        # Display Diagnostic Profiling Cards
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Records", f"{profile['total_rows']:,}")
        m2.metric("Missing Values", f"{profile['missing_count']:,}")
        m3.metric("Duplicates", f"{profile['duplicate_count']:,}")
        m4.metric("Avg Word Count", f"{profile['avg_word_count']:.1f}")
        m5.metric("Avg Character Length", f"{profile['avg_char_length']:.0f}")

        # Run Preprocessing Button / Automatic Preprocessing
        if st.button("Apply Preprocessing & Prepare Embeddings", type="primary"):
            with st.spinner("Sanitizing text and removing invalid records..."):
                proc_df, audit = preprocess_dataset(
                    raw_df,
                    query_col=query_col,
                    drop_duplicates=drop_dups,
                    min_chars=min_length
                )
                st.session_state.processed_df = proc_df
                st.session_state.audit_report = audit
                st.success(f"Retained {audit['records_retained']:,} high-quality queries ({audit['retention_rate_pct']}%)!")

        if st.session_state.processed_df is not None:
            audit = st.session_state.audit_report
            st.divider()
            st.markdown("#### Preprocessing Audit Report")
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Retained Queries", f"{audit['records_retained']:,}")
            a2.metric("Dropped Missing/Empty", f"{audit['dropped_missing'] + audit['dropped_empty']:,}")
            a3.metric("Dropped Too Short", f"{audit['dropped_too_short']:,}")
            a4.metric("Dropped Duplicates", f"{audit['dropped_duplicates']:,}")

            st.markdown("#### Sample Preview of Sanitized Queries")
            st.dataframe(
                st.session_state.processed_df[[query_col, "_cleaned_query"]].head(8),
                use_container_width=True
            )

# ----------------------------------------------------
# TAB 2: DISCOVERY ENGINE & CONFIGURATION
# ----------------------------------------------------
with tab_discovery:
    st.subheader("Intent Discovery Engine")
    
    if st.session_state.processed_df is None:
        st.warning("⚠️ Please complete data preprocessing in Tab 1 first.")
    else:
        proc_df = st.session_state.processed_df
        n_queries = len(proc_df)
        
        st.markdown(f"Ready to discover customer intents across **{n_queries:,} customer queries**.")
        
        col_algo, col_params = st.columns([1, 1])
        with col_algo:
            st.markdown("##### Clustering Strategy")
            algorithm_type = st.radio(
                "Discovery Algorithm:",
                [
                    "HDBSCAN (Automatic Intent Discovery with Noise Isolation)",
                    "Agglomerative Clustering (Hierarchical Cosine Linkage)",
                    "K-Means (Fixed Target Cluster Count)"
                ],
                help="HDBSCAN automatically infers the number of intents and isolates noise without forcing queries into arbitrary clusters."
            )
            
        with col_params:
            st.markdown("##### Hyperparameter Tuning")
            if "HDBSCAN" in algorithm_type:
                # Dynamic defaults based on dataset size
                default_min_size = max(2, min(10, n_queries // 15)) if n_queries < 200 else 10
                min_cluster_size = st.slider(
                    "Min Cluster Size (Minimum queries to form an intent)",
                    min_value=2,
                    max_value=max(3, min(100, n_queries // 2)),
                    value=default_min_size,
                    help="Smaller values discover fine-grained niche intents; larger values yield broad macro-intents."
                )
                min_samples = st.slider(
                    "Min Samples (Outlier Conservatism)",
                    min_value=1,
                    max_value=max(2, min_cluster_size),
                    value=max(1, min_cluster_size // 2),
                    help="Higher values classify more ambiguous queries as noise."
                )
                cluster_selection = st.selectbox("Cluster Selection Method", ["eom", "leaf"])
            elif "Agglomerative" in algorithm_type:
                auto_k = st.checkbox("Automatically optimize number of intents via Silhouette sweep", value=True)
                if not auto_k:
                    agg_k = st.slider("Target Number of Intents", min_value=2, max_value=min(40, max(3, n_queries - 1)), value=5)
                else:
                    agg_k = None
            else:
                kmeans_k = st.slider("Number of Clusters (K)", min_value=2, max_value=min(40, max(3, n_queries - 1)), value=8)

        st.divider()
        run_btn = st.button("🚀 Discover Customer Intents Now", type="primary", use_container_width=True)

        if run_btn:
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            # Step 1: Embeddings
            def update_emb_progress(pct, msg):
                progress_bar.progress(float(pct * 0.4))
                status_text.text(f"Step 1/4: {msg}")

            status_text.text("Step 1/4: Generating semantic embeddings...")
            texts = proc_df["_cleaned_query"].tolist()
            embeddings = compute_embeddings(
                texts,
                model_name=selected_model,
                batch_size=64,
                normalize=True,
                use_cache=True,
                progress_callback=update_emb_progress
            )
            st.session_state.embeddings = embeddings

            # Step 2: Clustering
            status_text.text("Step 2/4: Executing unsupervised clustering...")
            progress_bar.progress(0.45)
            
            if "HDBSCAN" in algorithm_type:
                labels, probs, meta = run_hdbscan_clustering(
                    embeddings,
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_method=cluster_selection,
                    use_dim_reduction=dim_reduction_toggle
                )
            elif "Agglomerative" in algorithm_type:
                labels, probs, meta = run_agglomerative_clustering(embeddings, n_clusters=agg_k)
            else:
                labels, probs, meta = run_kmeans_clustering(embeddings, n_clusters=kmeans_k)

            st.session_state.labels = labels
            st.session_state.probs = probs

            # Step 3: Interpretation & c-TF-IDF
            status_text.text("Step 3/4: Synthesizing intent names & extracting c-TF-IDF keywords...")
            progress_bar.progress(0.70)
            interp = interpret_discovered_clusters(texts, embeddings, labels, probs)
            interp["clustering_meta"] = meta
            st.session_state.discovery_results = interp

            # Step 4: 2D Projections for Visualization
            status_text.text("Step 4/4: Computing 2D interactive manifold coordinates...")
            progress_bar.progress(0.85)
            coords_umap, _ = compute_2d_projection(embeddings, method="umap")
            coords_pca, _ = compute_2d_projection(embeddings, method="pca")
            st.session_state.coords_2d_umap = coords_umap
            st.session_state.coords_2d_pca = coords_pca

            # Evaluation metrics
            coherence_map = {c: prof["semantic_coherence"] for c, prof in interp["cluster_profiles"].items()}
            eval_metrics = evaluate_intent_clustering(embeddings, labels, coherence_map)
            st.session_state.eval_metrics = eval_metrics

            progress_bar.progress(1.0)
            status_text.text("✅ Intent discovery complete!")
            st.balloons()
            st.success(f"Discovered **{interp['n_clusters']} distinct customer intents** across {n_queries:,} queries!")

# ----------------------------------------------------
# TAB 3: DISCOVERED INTENTS DASHBOARD
# ----------------------------------------------------
with tab_intents:
    st.subheader("Discovered Customer Intent Segments")
    
    if st.session_state.discovery_results is None:
        st.info("💡 Run the discovery engine in Tab 2 to see the discovered intent breakdown.")
    else:
        results = st.session_state.discovery_results
        profiles = results["cluster_profiles"]
        noise_prof = results.get("noise_profile")
        eval_metrics = st.session_state.eval_metrics
        proc_df = st.session_state.processed_df
        q_col = st.session_state.selected_query_col

        # High-Level Metric Tiles
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Discovered Intents", f"{results['n_clusters']}")
        k2.metric("Clustered Queries", f"{eval_metrics['clustered_samples']:,} ({eval_metrics['clustered_percentage']}%)")
        k3.metric("Noise / Outliers", f"{eval_metrics['noise_count']:,} ({eval_metrics['noise_percentage']}%)")
        k4.metric("Mean Coherence", f"{eval_metrics.get('mean_semantic_coherence', 0.0):.3f}" if eval_metrics.get('mean_semantic_coherence') else "N/A")
        sil = eval_metrics.get("silhouette_score")
        k5.metric("Silhouette Score", f"{sil:.3f}" if sil is not None else "N/A")

        st.divider()

        # Search and Filter
        f1, f2 = st.columns([2, 1])
        with f1:
            search_term = st.text_input("🔍 Search customer queries across all intents:", placeholder="Type a keyword (e.g. 'refund', 'card', 'password')...")
        with f2:
            sort_by = st.selectbox("Sort Intents By:", ["Volume (Largest First)", "Volume (Smallest First)", "Semantic Coherence"])

        sorted_clusters = list(profiles.keys())
        if sort_by == "Volume (Largest First)":
            sorted_clusters.sort(key=lambda c: profiles[c]["size"], reverse=True)
        elif sort_by == "Volume (Smallest First)":
            sorted_clusters.sort(key=lambda c: profiles[c]["size"])
        elif sort_by == "Semantic Coherence":
            sorted_clusters.sort(key=lambda c: profiles[c]["semantic_coherence"], reverse=True)

        # Discovered Intent Cards
        for c in sorted_clusters:
            prof = profiles[c]
            
            # Check if search filter matches
            matching_queries = []
            if search_term:
                clean_term = search_term.lower()
                matching_queries = [
                    q for q in proc_df.iloc[np.where(proc_df["_cleaned_query"].isin([x["query"] for x in prof["top_queries"]]))[0]][q_col].tolist()
                    if clean_term in q.lower()
                ]

            with st.expander(
                f"**Intent #{c}: {prof['intent_name']}** — {prof['size']:,} queries ({prof['percentage']}%) | Coherence: {prof['semantic_coherence']:.3f}",
                expanded=(c == sorted_clusters[0])
            ):
                col_info, col_chart = st.columns([3, 2])
                with col_info:
                    st.markdown(f"**Description:** {prof['description']}")
                    
                    st.markdown("**Representative Customer Voice (Centroid Medoid):**")
                    st.markdown(f"<div class='medoid-quote'>\"{prof['representative_query']}\"</div>", unsafe_allow_html=True)

                    st.markdown("**Distinctive c-TF-IDF Keyphrases:**")
                    kw_badges = "".join([f"<span class='badge-keyword'>{k[0]} ({k[1]:.2f})</span>" for k in prof['keywords'][:6]])
                    st.markdown(kw_badges, unsafe_allow_html=True)

                with col_chart:
                    fig_kw = plot_cluster_keywords(prof, top_k=6)
                    st.plotly_chart(fig_kw, use_container_width=True)

                st.markdown("**Discovered Queries in this Intent Segment:**")
                cluster_mask = (st.session_state.labels == c)
                cluster_queries_df = proc_df[cluster_mask][[q_col]].copy()
                cluster_queries_df = cluster_queries_df.rename(columns={q_col: "Customer Query"})
                
                if search_term:
                    cluster_queries_df = cluster_queries_df[
                        cluster_queries_df["Customer Query"].str.contains(search_term, case=False, na=False)
                    ]
                    st.caption(f"Showing {len(cluster_queries_df)} queries matching '{search_term}'")
                    
                st.dataframe(cluster_queries_df, use_container_width=True, height=220)

                # Export single intent CSV
                csv_single = cluster_queries_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    f"📥 Export Intent #{c} Queries (CSV)",
                    data=csv_single,
                    file_name=f"intent_{c}_{prof['intent_name'][:20].replace(' ', '_').lower()}.csv",
                    mime="text/csv",
                    key=f"dl_intent_{c}"
                )

                st.markdown("**Top Centroid-Aligned Queries (Medoid Neighbors):**")
                top_df = pd.DataFrame(prof["top_queries"]).rename(columns={"query": "Customer Query", "similarity": "Cosine Similarity to Centroid"})
                st.dataframe(top_df, use_container_width=True)

        # Unassigned / Noise Section
        if noise_prof and noise_prof["size"] > 0:
            with st.expander(f"⚠️ **Unassigned Queries / Outliers (Noise)** — {noise_prof['size']:,} queries ({noise_prof['percentage']}%)"):
                st.markdown(f"*{noise_prof['description']}*")
                noise_mask = (st.session_state.labels == -1)
                noise_df = proc_df[noise_mask][[q_col]].rename(columns={q_col: "Unassigned Query"})
                st.dataframe(noise_df, use_container_width=True, height=220)
                csv_noise = noise_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Export Unassigned Outliers (CSV)",
                    data=csv_noise,
                    file_name="unassigned_noise_queries.csv",
                    mime="text/csv",
                    key="dl_noise_queries"
                )

# ----------------------------------------------------
# TAB 4: VISUALIZATIONS
# ----------------------------------------------------
with tab_viz:
    st.subheader("Interactive Semantic Visualizations")
    
    if st.session_state.discovery_results is None or st.session_state.coords_2d_umap is None:
        st.info("💡 Run the discovery engine in Tab 2 to view interactive visual manifold charts.")
    else:
        results = st.session_state.discovery_results
        proc_df = st.session_state.processed_df
        q_col = st.session_state.selected_query_col
        queries = proc_df[q_col].tolist()
        
        eval_metrics = st.session_state.eval_metrics
        labels = st.session_state.labels
        
        v1, v2 = st.tabs(["🗺️ 2D Semantic Cluster Map", "📊 Intent Volume & Frequency"])
        
        with v1:
            proj_mode = st.radio("Manifold Projection:", ["UMAP (Non-linear Semantic)", "PCA (Linear Variance)"], horizontal=True)
            coords = st.session_state.coords_2d_umap if "UMAP" in proj_mode else st.session_state.coords_2d_pca
            labels = st.session_state.labels
            
            fig_map = plot_cluster_map_2d(
                coords,
                queries,
                labels,
                results["cluster_profiles"],
                projection_name="UMAP" if "UMAP" in proj_mode else "PCA"
            )
            # Render interactive 2D scatter
            st.plotly_chart(fig_map, use_container_width=True)
            st.caption("Hover over any data point to inspect the authentic customer query, assigned intent, and semantic coordinates.")

        with v2:
            fig_bar = plot_intent_size_distribution(
                results["cluster_profiles"],
                noise_count=eval_metrics.get("noise_count", 0)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

# ----------------------------------------------------
# TAB 5: QUALITY EVALUATION & OVERLAP
# ----------------------------------------------------
with tab_eval:
    st.subheader("Clustering Quality & Semantic Overlap Diagnostics")
    
    if st.session_state.eval_metrics is None:
        st.info("💡 Run the discovery engine in Tab 2 to calculate clustering evaluation metrics.")
    else:
        ev = st.session_state.eval_metrics
        res = st.session_state.discovery_results
        
        # Metric Diagnostics Table
        st.markdown("#### Primary Unsupervised Validation Metrics")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Silhouette Score</div>
                <div class='metric-val'>{ev.get('silhouette_score') if ev.get('silhouette_score') is not None else 'N/A'}</div>
                <div style='font-size:0.8rem; color:#64748b; margin-top:6px;'>
                    <b>Interpretation:</b> {ev.get('interpretations', {}).get('silhouette', ev.get('silhouette_status', ''))}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Davies-Bouldin Index (DBI)</div>
                <div class='metric-val'>{ev.get('davies_bouldin_index') if ev.get('davies_bouldin_index') is not None else 'N/A'}</div>
                <div style='font-size:0.8rem; color:#64748b; margin-top:6px;'>
                    <b>Interpretation:</b> {ev.get('interpretations', {}).get('davies_bouldin', ev.get('davies_bouldin_status', ''))}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with c3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Calinski-Harabasz Index (CHI)</div>
                <div class='metric-val'>{ev.get('calinski_harabasz_index') if ev.get('calinski_harabasz_index') is not None else 'N/A'}</div>
                <div style='font-size:0.8rem; color:#64748b; margin-top:6px;'>
                    <b>Interpretation:</b> Higher ratio indicates tighter, better-separated clusters.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # Semantic Overlap Matrix & Merge Recommendations
        st.markdown("#### Intent Semantic Overlap Matrix (Centroid Cosine Proximity)")
        fig_heat = plot_intercluster_similarity_heatmap(res["cluster_profiles"])
        if fig_heat is not None:
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.write("Need at least 2 discovered intent clusters to compute inter-intent similarity.")

        # Flagged Overlap Pairs
        overlaps = res.get("overlap_pairs", [])
        if overlaps:
            st.warning(f"⚠️ Found {len(overlaps)} intent pair(s) with high semantic similarity (≥ 0.70). Review for potential consolidation:")
            st.dataframe(pd.DataFrame(overlaps), use_container_width=True)
        else:
            st.success("✅ No high semantic overlaps detected across discovered intent clusters (all pairwise similarities < 0.70).")

# ----------------------------------------------------
# TAB 6: EXPORT CENTER
# ----------------------------------------------------
with tab_export:
    st.subheader("Data Export Center")
    
    if st.session_state.discovery_results is None:
        st.info("💡 Run the discovery engine in Tab 2 to generate downloadable exports.")
    else:
        res = st.session_state.discovery_results
        proc_df = st.session_state.processed_df
        q_col = st.session_state.selected_query_col
        eval_metrics = st.session_state.eval_metrics
        audit = st.session_state.audit_report

        # Create Annotated DataFrame
        annotated_df = create_annotated_dataframe(
            proc_df,
            labels=st.session_state.labels,
            probabilities=st.session_state.probs,
            cluster_profiles=res["cluster_profiles"],
            original_query_col=q_col
        )
        
        summary_df = create_intent_summary_dataframe(
            res["cluster_profiles"],
            noise_profile=res.get("noise_profile")
        )

        col_e1, col_e2, col_e3 = st.columns(3)
        with col_e1:
            st.markdown("##### 📄 Annotated Dataset (CSV)")
            st.caption("Original customer records enriched with discovered intent ID, intent name, description, confidence, and top keywords.")
            csv_annotated = annotated_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Annotated Queries (CSV)",
                data=csv_annotated,
                file_name="discovered_customer_intents_annotated.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True
            )

        with col_e2:
            st.markdown("##### 📊 Intent Summary (CSV)")
            st.caption("Executive overview of all discovered intents, query volumes, percentages, medoids, and coherence scores.")
            csv_summary = summary_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Intent Summary (CSV)",
                data=csv_summary,
                file_name="intent_discovery_summary.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col_e3:
            st.markdown("##### ⚡ Full Analysis Schema (JSON)")
            st.caption("Complete programmatic payload including preprocessing audit, clustering metadata, evaluation metrics, and intent profiles.")
            json_str = create_json_export(
                res["cluster_profiles"],
                eval_metrics,
                audit,
                noise_profile=res.get("noise_profile")
            )
            st.download_button(
                "Download Full Results (JSON)",
                data=json_str,
                file_name="intent_discovery_full_results.json",
                mime="application/json",
                use_container_width=True
            )

        st.divider()
        st.markdown("#### Preview of Annotated Output Data")
        st.dataframe(annotated_df.head(10), use_container_width=True)

# ----------------------------------------------------
# TAB 7: METHODOLOGY & DOCUMENTATION
# ----------------------------------------------------
with tab_docs:
    st.subheader("Platform Methodology & Algorithmic Foundation")
    st.markdown("""
    ### How It Works

    #### 1. Contextual Embeddings
    We utilize `sentence-transformers/all-MiniLM-L6-v2` to map short, conversational customer queries into a dense 384-dimensional semantic vector space. Unlike traditional TF-IDF, embeddings capture semantic equivalence:
    * `"I lost my wallet and credit card"` and `"Can you block my visa card immediately"` are mapped close together despite having few shared words.

    #### 2. Manifold Learning & Dimension Reduction
    High-dimensional vector spaces suffer from metric concentration (the *curse of dimensionality*). We employ **UMAP** (Uniform Manifold Approximation and Projection) with cosine metric to reduce 384 dimensions to a 5D manifold. This densifies natural clusters and dramatically improves density-based clustering accuracy.

    #### 3. Automatic Density Clustering with HDBSCAN
    Unlike K-Means (which assumes spherical clusters of equal size and forces every outlier into a cluster), **HDBSCAN**:
    - Discovers the natural number of intents without requiring user guesswork.
    - Accommodates variable densities (common high-frequency intents vs. niche long-tail intents).
    - Automatically isolates ambiguous, garbled, or unique queries into an unassigned **Noise (-1)** category.

    #### 4. Automatic Labeling with c-TF-IDF & Centroid Medoid
    - **c-TF-IDF (Class-based TF-IDF)** calculates which words and n-grams appear frequently within an intent group relative to all other intents, pinpointing distinguishing keywords.
    - The **Centroid Medoid** identifies the actual customer query closest to the mathematical center of the cluster. This guarantees 100% authentic customer voice without generative hallucinations.

    #### 5. Rigorous Evaluation
    - **Silhouette Score**: Evaluates cluster cohesion vs. separation across valid clusters.
    - **Davies-Bouldin Index (DBI)**: Measures average similarity between clusters (lower is better).
    - **Semantic Coherence**: Measures mean intra-cluster cosine similarity to the centroid.
    - **Centroid Similarity Matrix**: Flags potential overlapping intents (similarity &ge; 0.70) for merge consideration.
    """)
