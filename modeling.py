from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ACTIVITY_WEIGHTS = {
    "view": 1,
    "search": 2,
    "add_to_cart": 5,
}
N_CLUSTERS = 10
FEATURE_COLUMNS = ["category_id", "brand_id", "shoe_type_id"]


def build_product_feature_matrix(products_df: pd.DataFrame):
    normalized = products_df[FEATURE_COLUMNS].copy()
    for column in FEATURE_COLUMNS:
        normalized[column] = normalized[column].fillna("__missing__").astype("string")
    features = pd.get_dummies(normalized, columns=FEATURE_COLUMNS, dtype=float)
    column_groups = {
        field: tuple(column for column in features.columns if column.startswith(f"{field}_"))
        for field in FEATURE_COLUMNS
    }
    return features, column_groups


def cluster_products(feature_matrix, n_clusters: int):
    if len(feature_matrix) == 0:
        return np.array([], dtype=int), None
    if len(feature_matrix) == 1:
        labels = np.array([0], dtype=int)
        labels.setflags(write=False)
        return labels, None

    distinct_points = np.unique(feature_matrix, axis=0).shape[0]
    k = max(1, min(n_clusters, len(feature_matrix), distinct_points))
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = model.fit_predict(feature_matrix)
    cluster_labels = np.asarray(cluster_labels, dtype=int)
    cluster_labels.setflags(write=False)
    return cluster_labels, model


def weighted_activity_strength(activity_df: pd.DataFrame):
    if "activity_count" in activity_df:
        counts = pd.to_numeric(activity_df["activity_count"], errors="coerce").fillna(1).clip(lower=1)
    else:
        counts = 1
    weights = activity_df["activity_type"].map(ACTIVITY_WEIGHTS).fillna(0)
    return weights * counts


def get_user_dominant_clusters(activity_df, product_ids, cluster_labels, top_n: int = 2):
    activity_df = activity_df.copy()
    activity_df["weight"] = weighted_activity_strength(activity_df)

    product_to_cluster = dict(zip(product_ids, cluster_labels))
    activity_df["cluster"] = activity_df["product_id"].map(product_to_cluster)
    activity_df = activity_df.dropna(subset=["cluster"])

    if activity_df.empty:
        return []

    cluster_scores = activity_df.groupby("cluster")["weight"].sum().sort_values(ascending=False)
    return [int(cluster) for cluster in cluster_scores.head(top_n).index]


def build_user_preference_vector(feature_matrix, product_ids, activity_df):
    activity_df_weighted = activity_df.copy()
    activity_df_weighted["weight"] = weighted_activity_strength(activity_df_weighted)
    product_weights = activity_df_weighted.groupby("product_id")["weight"].sum()
    weights_aligned = np.array([product_weights.get(product_id, 0) for product_id in product_ids])

    if weights_aligned.sum() == 0:
        return np.zeros(feature_matrix.shape[1])

    return (feature_matrix.T @ weights_aligned) / weights_aligned.sum()


def rank_candidates_by_tier(candidates_df, feature_matrix, column_groups, user_vector):
    brand_idx = [feature_matrix.columns.get_loc(column) for column in column_groups["brand_id"]]
    shoe_type_idx = [feature_matrix.columns.get_loc(column) for column in column_groups["shoe_type_id"]]
    category_idx = [feature_matrix.columns.get_loc(column) for column in column_groups["category_id"]]
    candidate_vectors = feature_matrix.values[candidates_df.index]

    ranked = candidates_df.copy()
    ranked["brand_score"] = candidate_vectors[:, brand_idx] @ user_vector[brand_idx]
    ranked["shoe_type_score"] = candidate_vectors[:, shoe_type_idx] @ user_vector[shoe_type_idx]
    ranked["category_score"] = candidate_vectors[:, category_idx] @ user_vector[category_idx]
    return ranked.sort_values(
        by=["brand_score", "shoe_type_score", "category_score"],
        ascending=[False, False, False],
    )
