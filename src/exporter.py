"""
Data Export and Serialization Engine for Customer Intent Discovery Platform.

Provides:
- Fully annotated CSV with assigned intent IDs, names, descriptions, and confidence
- Executive Summary CSV of all discovered intent segments
- Full JSON schema export of metrics, metadata, and cluster parameters
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import json


def create_annotated_dataframe(
    df_processed: pd.DataFrame,
    labels: List[int],
    probabilities: List[float],
    cluster_profiles: Dict[int, Dict[str, Any]],
    original_query_col: str
) -> pd.DataFrame:
    """
    Merge discovered intents, metadata, and labels back into the original dataset structure.
    """
    export_df = df_processed.copy()
    export_df["discovered_intent_id"] = labels
    export_df["clustering_confidence"] = [round(float(p), 4) for p in probabilities]
    export_df["is_noise_outlier"] = [bool(l == -1) for l in labels]

    intent_names = []
    intent_descs = []
    intent_kws = []

    for l in labels:
        if l == -1:
            intent_names.append("Unassigned / Outlier")
            intent_descs.append("Outlier query that does not belong to any discovered cluster.")
            intent_kws.append("")
        else:
            prof = cluster_profiles.get(l, {})
            intent_names.append(prof.get("intent_name", f"Intent {l}"))
            intent_descs.append(prof.get("description", ""))
            kws = prof.get("keywords", [])
            kw_str = ", ".join([k[0] for k in kws[:5]])
            intent_kws.append(kw_str)

    export_df["discovered_intent_name"] = intent_names
    export_df["discovered_intent_description"] = intent_descs
    export_df["intent_top_keywords"] = intent_kws

    # Clean internal helper columns
    cols_to_drop = [c for c in ["_cleaned_query"] if c in export_df.columns]
    if cols_to_drop:
        export_df = export_df.drop(columns=cols_to_drop)

    return export_df


def create_intent_summary_dataframe(
    cluster_profiles: Dict[int, Dict[str, Any]],
    noise_profile: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Generate clean tabular summary of all discovered intents.
    """
    rows = []
    for c, prof in sorted(cluster_profiles.items()):
        kw_str = ", ".join([f"{k[0]} ({k[1]:.2f})" for k in prof.get("keywords", [])[:5]])
        rows.append({
            "intent_id": c,
            "intent_name": prof.get("intent_name", ""),
            "description": prof.get("description", ""),
            "representative_customer_query": prof.get("representative_query", ""),
            "query_count": prof.get("size", 0),
            "percentage_of_total": prof.get("percentage", 0.0),
            "semantic_coherence": prof.get("semantic_coherence", 0.0),
            "top_keywords": kw_str
        })

    if noise_profile and noise_profile.get("size", 0) > 0:
        rows.append({
            "intent_id": -1,
            "intent_name": noise_profile.get("intent_name", "Unassigned"),
            "description": noise_profile.get("description", ""),
            "representative_customer_query": noise_profile.get("representative_query", ""),
            "query_count": noise_profile.get("size", 0),
            "percentage_of_total": noise_profile.get("percentage", 0.0),
            "semantic_coherence": 0.0,
            "top_keywords": "N/A"
        })

    return pd.DataFrame(rows)


def create_json_export(
    cluster_profiles: Dict[int, Dict[str, Any]],
    evaluation_metrics: Dict[str, Any],
    audit_report: Dict[str, Any],
    noise_profile: Optional[Dict[str, Any]] = None
) -> str:
    """
    Serialize discovery results and evaluation to JSON string.
    """
    # Clean out non-serializable objects (like numpy arrays)
    clean_profiles = {}
    for c, p in cluster_profiles.items():
        cp = p.copy()
        if "centroid" in cp:
            del cp["centroid"]
        clean_profiles[str(c)] = cp

    payload = {
        "metadata": {
            "platform": "Customer Intent Discovery Platform",
            "version": "1.0.0",
        },
        "data_audit": audit_report,
        "evaluation_metrics": evaluation_metrics,
        "discovered_intents": clean_profiles,
        "noise_segment": noise_profile
    }
    return json.dumps(payload, indent=2, default=str)
