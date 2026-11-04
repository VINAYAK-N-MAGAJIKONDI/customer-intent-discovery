"""
Interactive Visualizations for Customer Intent Discovery Platform.

Builds high-resolution, responsive Plotly charts:
- 2D Semantic Cluster Map (UMAP & PCA) with rich hover inspection
- Intent Size & Frequency Distribution
- Top Keywords by Intent (c-TF-IDF bar plots)
- Inter-Cluster Centroid Similarity Heatmap (Overlap Explorer)
- Noise & Outlier breakdown
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_cluster_map_2d(
    coords_2d: np.ndarray,
    queries: List[str],
    labels: np.ndarray,
    cluster_profiles: Dict[int, Dict[str, Any]],
    projection_name: str = "UMAP"
) -> go.Figure:
    """
    Build interactive 2D scatter plot with query inspection tooltips.
    """
    n_samples = len(queries)
    plot_data = []

    # Palette
    color_discrete_sequence = px.colors.qualitative.Dark24 + px.colors.qualitative.Alphabet

    for i in range(n_samples):
        label = labels[i]
        q = queries[i]
        
        # Truncate for hover
        hover_q = q if len(q) <= 90 else q[:87] + "..."

        if label == -1:
            intent_name = "Noise / Outlier"
            category_color = "Outliers"
        else:
            profile = cluster_profiles.get(label, {})
            intent_name = f"Cluster {label}: {profile.get('intent_name', 'Intent')}"
            category_color = intent_name

        plot_data.append({
            "x": float(coords_2d[i, 0]),
            "y": float(coords_2d[i, 1]),
            "query": hover_q,
            "full_query": q,
            "cluster_id": label,
            "intent_name": intent_name,
            "is_noise": (label == -1)
        })

    df_plot = pd.DataFrame(plot_data)

    fig = px.scatter(
        df_plot,
        x="x",
        y="y",
        color="intent_name",
        hover_name="intent_name",
        hover_data={"query": True, "x": False, "y": False, "intent_name": False},
        color_discrete_sequence=color_discrete_sequence,
        template="plotly_white",
        labels={"x": f"{projection_name} Component 1", "y": f"{projection_name} Component 2"}
    )

    # Style noise queries distinctly (grey transparent points)
    fig.for_each_trace(
        lambda trace: trace.update(marker=dict(size=5, opacity=0.35, color="#94a3b8"))
        if "Noise" in trace.name else trace.update(marker=dict(size=7, opacity=0.85))
    )

    fig.update_layout(
        title={
            "text": f"<b>Interactive Semantic Intent Map</b> ({projection_name} 2D Manifold)",
            "y": 0.96,
            "x": 0.02,
            "xanchor": "left",
            "yanchor": "top"
        },
        legend_title_text="Discovered Intents",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.35,
            xanchor="center",
            x=0.5,
            font=dict(size=10)
        ),
        margin=dict(l=20, r=20, t=50, b=80),
        height=620,
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="sans-serif")
    )
    return fig


def plot_intent_size_distribution(
    cluster_profiles: Dict[int, Dict[str, Any]],
    noise_count: int = 0
) -> go.Figure:
    """
    Horizontal bar chart showing query volume per discovered intent.
    """
    records = []
    for c, prof in cluster_profiles.items():
        records.append({
            "intent_label": f"#{c}: {prof['intent_name']}",
            "size": prof["size"],
            "percentage": prof["percentage"],
            "type": "Discovered Intent"
        })

    if noise_count > 0:
        total = sum(r["size"] for r in records) + noise_count
        records.append({
            "intent_label": "Unassigned / Noise",
            "size": noise_count,
            "percentage": round((noise_count / total) * 100, 2),
            "type": "Noise / Outlier"
        })

    df = pd.DataFrame(records).sort_values("size", ascending=True)

    fig = px.bar(
        df,
        x="size",
        y="intent_label",
        orientation="h",
        color="type",
        color_discrete_map={"Discovered Intent": "#2563eb", "Noise / Outlier": "#94a3b8"},
        text="size",
        labels={"size": "Number of Queries", "intent_label": "Intent Group"},
        template="plotly_white"
    )

    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        title="<b>Query Volume per Discovered Intent</b>",
        margin=dict(l=20, r=40, t=50, b=30),
        height=max(360, len(records) * 28 + 100),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_cluster_keywords(
    cluster_profile: Dict[str, Any],
    top_k: int = 8
) -> go.Figure:
    """
    Bar chart showing top c-TF-IDF keyword weights for an individual intent.
    """
    kws = cluster_profile.get("keywords", [])[:top_k]
    if not kws:
        fig = go.Figure()
        fig.update_layout(title="No distinct keywords available.")
        return fig

    df = pd.DataFrame(kws, columns=["Keyword", "Score"]).sort_values("Score", ascending=True)

    fig = px.bar(
        df,
        x="Score",
        y="Keyword",
        orientation="h",
        text="Score",
        template="plotly_white",
        color="Score",
        color_continuous_scale="Blues"
    )

    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", cliponaxis=False)
    fig.update_layout(
        title=f"<b>Distinctive Keywords (c-TF-IDF): {cluster_profile.get('intent_name', '')}</b>",
        margin=dict(l=20, r=40, t=50, b=30),
        height=320,
        coloraxis_showscale=False
    )
    return fig


def plot_intercluster_similarity_heatmap(
    cluster_profiles: Dict[int, Dict[str, Any]]
) -> Optional[go.Figure]:
    """
    Heatmap displaying cosine similarity between all discovered intent centroids.
    Highlights overlapping or semantically similar clusters.
    """
    cluster_ids = sorted(cluster_profiles.keys())
    if len(cluster_ids) < 2:
        return None

    names = [f"#{c}: {cluster_profiles[c]['intent_name'][:20]}" for c in cluster_ids]
    centroids = np.array([cluster_profiles[c]["centroid"] for c in cluster_ids])

    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(centroids)

    fig = px.imshow(
        sim_matrix,
        x=names,
        y=names,
        color_continuous_scale="RdBu_r",
        zmin=0.0,
        zmax=1.0,
        labels=dict(color="Cosine Similarity"),
        text_auto=".2f",
        template="plotly_white"
    )

    fig.update_layout(
        title="<b>Inter-Intent Semantic Overlap Heatmap</b> (Centroid Cosine Similarity)",
        margin=dict(l=40, r=40, t=50, b=40),
        height=max(420, len(cluster_ids) * 28 + 140)
    )
    return fig
