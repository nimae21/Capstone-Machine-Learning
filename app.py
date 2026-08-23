from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from db import fetch_all

app = Flask(__name__)

ACTIVITY_WEIGHTS = {
    'view': 1,
    'search': 2,
    'add_to_cart': 5,
}

# Number of clusters K-Means will discover. This is a hyperparameter —
# chosen based on catalog size. Rule of thumb: enough clusters that
# products meaningfully separate, not so many that clusters become tiny.
N_CLUSTERS = 5


def build_product_feature_matrix(products_df):
    """One-hot encode each product into a numeric vector space."""
    features = pd.get_dummies(
        products_df[['category_id', 'brand_id', 'shoe_type_id']],
        columns=['category_id', 'brand_id', 'shoe_type_id']
    )
    return features


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
    k = min(n_clusters, len(feature_matrix))  # can't have more clusters than products
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = model.fit_predict(feature_matrix)
    return cluster_labels, model


def get_user_dominant_cluster(activity_df, product_ids, cluster_labels):
    """
    Determine which cluster the user gravitates toward, weighted by
    how strongly they interacted with products in each cluster.
    """
    activity_df = activity_df.copy()
    activity_df['weight'] = activity_df['activity_type'].map(ACTIVITY_WEIGHTS)

    product_to_cluster = dict(zip(product_ids, cluster_labels))
    activity_df['cluster'] = activity_df['product_id'].map(product_to_cluster)
    activity_df = activity_df.dropna(subset=['cluster'])

    if activity_df.empty:
        return None

    cluster_scores = activity_df.groupby('cluster')['weight'].sum()
    return int(cluster_scores.idxmax())


@app.route('/recommendations/<int:user_id>', methods=['GET'])
def get_recommendations(user_id):
    limit = int(request.args.get('limit', 8))

    activities = fetch_all("""
        SELECT ua.product_id, ua.activity_type
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

    feature_matrix = build_product_feature_matrix(products_df)

    # --- UNSUPERVISED LEARNING: discover product clusters ---
    cluster_labels, kmeans_model = cluster_products(feature_matrix.values, N_CLUSTERS)
    products_df['cluster'] = cluster_labels

    # Determine which cluster this user prefers, based on their activity
    dominant_cluster = get_user_dominant_cluster(activity_df, product_ids, cluster_labels)

    if dominant_cluster is None:
        return jsonify({'product_ids': [], 'reason': 'no_cluster_signal'})

    # Candidate products: same cluster, not already seen
    candidates = products_df[
        (products_df['cluster'] == dominant_cluster) &
        (~products_df['product_id'].isin(already_seen_ids))
    ]

    if candidates.empty:
        return jsonify({'product_ids': [], 'reason': 'cluster_exhausted'})

    # Rank within the cluster using cosine similarity to the user's
    # weighted preference vector, so results aren't just "same cluster,
    # arbitrary order" but genuinely ranked by closeness of fit.
    activity_df_weighted = activity_df.copy()
    activity_df_weighted['weight'] = activity_df_weighted['activity_type'].map(ACTIVITY_WEIGHTS)
    product_weights = activity_df_weighted.groupby('product_id')['weight'].sum()
    weights_aligned = np.array([product_weights.get(pid, 0) for pid in product_ids])

    user_vector = (feature_matrix.values.T @ weights_aligned) / weights_aligned.sum()
    user_vector = user_vector.reshape(1, -1)

    candidate_indices = candidates.index
    candidate_vectors = feature_matrix.values[candidate_indices]
    similarities = cosine_similarity(user_vector, candidate_vectors)[0]

    candidates = candidates.copy()
    candidates['similarity'] = similarities
    candidates = candidates.sort_values('similarity', ascending=False)

    top_products = candidates.head(limit)['product_id'].tolist()

    return jsonify({
        'product_ids': top_products,
        'reason': 'unsupervised_clustering',
        'cluster_assigned': dominant_cluster,
        'total_clusters': int(kmeans_model.n_clusters),
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)