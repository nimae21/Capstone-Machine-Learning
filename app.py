import hmac
import os

from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from db import fetch_all

app = Flask(__name__)
RECOMMENDATION_SERVICE_KEY = os.environ.get('RECOMMENDATION_SERVICE_KEY', '')

ACTIVITY_WEIGHTS = {
    'view': 1,
    'search': 2,
    'add_to_cart': 5,
}

# Number of clusters K-Means will discover. Raised from 5 to 10 so
# clusters can separate on brand identity as well as shoe type/category,
# rather than being forced to merge different brands together just to
# hit a small cluster count.
N_CLUSTERS = 10

FEATURE_COLUMNS = ['category_id', 'brand_id', 'shoe_type_id']


def build_product_feature_matrix(products_df):
    """
    One-hot encode each product into a numeric vector space, keeping
    track of which columns belong to which original field so we can
    later score candidates tier-by-tier (brand, then shoe_type, then
    category) instead of treating every feature as equally important.
    """
    features = pd.get_dummies(products_df[FEATURE_COLUMNS], columns=FEATURE_COLUMNS)

    column_groups = {
        field: [c for c in features.columns if c.startswith(f"{field}_")]
        for field in FEATURE_COLUMNS
    }

    return features, column_groups


def cluster_products(feature_matrix, n_clusters):
    """
    UNSUPERVISED LEARNING STEP.
    K-Means receives only the feature vectors — no labels, no target
    variable, no 'correct' grouping. It discovers product groupings
    purely from the structure of the data itself (which products have
    similar category/brand/shoe_type patterns), by iteratively:
      1. Placing K cluster centroids
      2. Assigning each product to its nearest centroid
      3. Recomputing centroids as the mean of assigned products
      4. Repeating until assignments stabilize (convergence)
    """
    distinct_points = np.unique(feature_matrix, axis=0).shape[0]
    k = min(n_clusters, len(feature_matrix), distinct_points)
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = model.fit_predict(feature_matrix)
    return cluster_labels, model


def get_user_dominant_clusters(activity_df, product_ids, cluster_labels, top_n=2):
    """
    Determine which cluster(s) the user gravitates toward, weighted by
    how strongly they interacted with products in each cluster. Returns
    up to top_n clusters so a user with genuinely split interests (e.g.
    both basketball and running shoes) isn't forced into a single
    winner-take-all cluster.
    """
    activity_df = activity_df.copy()
    activity_df['weight'] = weighted_activity_strength(activity_df)

    product_to_cluster = dict(zip(product_ids, cluster_labels))
    activity_df['cluster'] = activity_df['product_id'].map(product_to_cluster)
    activity_df = activity_df.dropna(subset=['cluster'])

    if activity_df.empty:
        return []

    cluster_scores = activity_df.groupby('cluster')['weight'].sum().sort_values(ascending=False)
    return [int(c) for c in cluster_scores.head(top_n).index]


def build_user_preference_vector(feature_matrix, product_ids, activity_df):
    """
    Weighted average of the feature vectors of every product the user
    interacted with, weighted by activity strength (view/search/cart).
    """
    activity_df_weighted = activity_df.copy()
    activity_df_weighted['weight'] = weighted_activity_strength(activity_df_weighted)
    product_weights = activity_df_weighted.groupby('product_id')['weight'].sum()
    weights_aligned = np.array([product_weights.get(pid, 0) for pid in product_ids])

    if weights_aligned.sum() == 0:
        return np.zeros(feature_matrix.shape[1])

    return (feature_matrix.T @ weights_aligned) / weights_aligned.sum()


def weighted_activity_strength(activity_df):
    """Apply the existing event weights and the number of events in each aggregate."""
    if 'activity_count' in activity_df:
        counts = pd.to_numeric(activity_df['activity_count'], errors='coerce').fillna(1).clip(lower=1)
    else:
        counts = 1

    return activity_df['activity_type'].map(ACTIVITY_WEIGHTS) * counts


def rank_candidates_by_tier(candidates_df, feature_matrix, column_groups, user_vector):
    """
    Score each candidate against the user's preference vector separately
    per feature group (brand, shoe_type, category), then sort so that a
    brand match always outranks a shoe_type-only match, which always
    outranks a category-only match — brand > shoe_type > category, in
    that strict order. Within the same tier, ties are broken by overall
    closeness (sum of the three scores).

    This keeps genuine unsupervised clustering as the mechanism that
    selects the *candidate pool* (see cluster_products), while giving
    predictable, explainable ordering on how that pool gets ranked.
    """
    brand_cols = column_groups['brand_id']
    shoe_type_cols = column_groups['shoe_type_id']
    category_cols = column_groups['category_id']

    brand_idx = [feature_matrix.columns.get_loc(c) for c in brand_cols]
    shoe_type_idx = [feature_matrix.columns.get_loc(c) for c in shoe_type_cols]
    category_idx = [feature_matrix.columns.get_loc(c) for c in category_cols]

    candidate_vectors = feature_matrix.values[candidates_df.index]

    brand_scores = candidate_vectors[:, brand_idx] @ user_vector[brand_idx]
    shoe_type_scores = candidate_vectors[:, shoe_type_idx] @ user_vector[shoe_type_idx]
    category_scores = candidate_vectors[:, category_idx] @ user_vector[category_idx]

    ranked = candidates_df.copy()
    ranked['brand_score'] = brand_scores
    ranked['shoe_type_score'] = shoe_type_scores
    ranked['category_score'] = category_scores

    ranked = ranked.sort_values(
        by=['brand_score', 'shoe_type_score', 'category_score'],
        ascending=[False, False, False],
    )

    return ranked


@app.route('/recommendations/<int:user_id>', methods=['GET'])
def get_recommendations(user_id):
    if user_id < 1:
        return jsonify({'error': 'invalid_user'}), 422
    if not RECOMMENDATION_SERVICE_KEY:
        return jsonify({'error': 'service_not_configured'}), 503

    supplied_key = request.headers.get('X-Recommendation-Key', '')
    if not hmac.compare_digest(supplied_key, RECOMMENDATION_SERVICE_KEY):
        return jsonify({'error': 'unauthorized'}), 401

    try:
        limit = int(request.args.get('limit', 8))
    except (TypeError, ValueError):
        return jsonify({'error': 'invalid_limit'}), 422
    if not 1 <= limit <= 20:
        return jsonify({'error': 'invalid_limit'}), 422

    activities = fetch_all("""
        SELECT ua.product_id, ua.activity_type, COALESCE(ua.activity_count, 1) AS activity_count
        FROM user_activities ua
        WHERE ua.user_id = %s
    """, (user_id,))

    if not activities:
        return jsonify({'product_ids': [], 'reason': 'no_activity_history'})

    activity_df = pd.DataFrame(activities)
    already_seen_ids = set(activity_df['product_id'].unique())

    all_products = fetch_all("""
        SELECT product_id, category_id, brand_id, shoe_type_id
        FROM products
        WHERE is_active = true
    """)

    if len(all_products) < 2:
        return jsonify({'product_ids': [], 'reason': 'insufficient_catalog'})

    products_df = pd.DataFrame(all_products)
    product_ids = products_df['product_id'].tolist()

    feature_matrix, column_groups = build_product_feature_matrix(products_df)

    # --- UNSUPERVISED LEARNING: discover product clusters ---
    cluster_labels, kmeans_model = cluster_products(feature_matrix.values, N_CLUSTERS)
    products_df['cluster'] = cluster_labels

    # Determine which cluster(s) this user prefers, based on their activity
    dominant_clusters = get_user_dominant_clusters(activity_df, product_ids, cluster_labels, top_n=2)

    if not dominant_clusters:
        return jsonify({'product_ids': [], 'reason': 'no_cluster_signal'})

    # Prefer products from the user's cluster(s), but keep recommendations
    # available when those clusters contain only products already seen.
    cluster_candidates = products_df[
        (products_df['cluster'].isin(dominant_clusters)) &
        (~products_df['product_id'].isin(already_seen_ids))
    ]

    candidates = cluster_candidates
    if candidates.empty:
        candidates = products_df[~products_df['product_id'].isin(already_seen_ids)]

    if candidates.empty:
        return jsonify({'product_ids': [], 'reason': 'catalog_exhausted'})

    # Rank candidates: brand match first, then shoe_type match, then
    # category match — so a same-brand product always outranks a
    # same-shoe-type-different-brand product, which always outranks a
    # same-category-only product. Other brands still appear, just lower.
    user_vector = build_user_preference_vector(feature_matrix.values, product_ids, activity_df)
    candidates = rank_candidates_by_tier(candidates, feature_matrix, column_groups, user_vector)

    top_products = candidates.head(limit)['product_id'].tolist()

    return jsonify({
        'product_ids': top_products,
        'reason': 'unsupervised_clustering',
        'clusters_assigned': dominant_clusters,
        'total_clusters': int(kmeans_model.n_clusters),
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', '5000'))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
