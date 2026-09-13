import unittest

import numpy as np
import pandas as pd

import app as recommendation_app


class AggregatedActivityWeightTest(unittest.TestCase):
    def test_counts_multiply_existing_activity_weights_for_clusters(self):
        activities = pd.DataFrame([
            {'product_id': 10, 'activity_type': 'view', 'activity_count': 6},
            {'product_id': 20, 'activity_type': 'add_to_cart', 'activity_count': 1},
        ])

        clusters = recommendation_app.get_user_dominant_clusters(
            activities,
            [10, 20],
            np.array([0, 1]),
        )

        self.assertEqual([0, 1], clusters)

    def test_counts_multiply_existing_activity_weights_for_preferences(self):
        activities = pd.DataFrame([
            {'product_id': 10, 'activity_type': 'view', 'activity_count': 5},
            {'product_id': 20, 'activity_type': 'add_to_cart', 'activity_count': 1},
        ])
        features = np.array([
            [1, 0],
            [0, 1],
        ])

        preference = recommendation_app.build_user_preference_vector(
            features,
            [10, 20],
            activities,
        )

        np.testing.assert_array_equal(np.array([0.5, 0.5]), preference)

    def test_legacy_unaggregated_rows_still_count_once(self):
        activities = pd.DataFrame([
            {'product_id': 10, 'activity_type': 'search'},
        ])

        self.assertEqual(
            [recommendation_app.ACTIVITY_WEIGHTS['search']],
            recommendation_app.weighted_activity_strength(activities).tolist(),
        )


if __name__ == '__main__':
    unittest.main()
