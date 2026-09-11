import unittest
from unittest.mock import patch

import app as recommendation_app


class RecommendationSecurityTest(unittest.TestCase):
    def setUp(self):
        self.client = recommendation_app.app.test_client()
        recommendation_app.RECOMMENDATION_SERVICE_KEY = 'shared-test-key'

    def test_missing_service_configuration_fails_closed(self):
        recommendation_app.RECOMMENDATION_SERVICE_KEY = ''
        response = self.client.get('/recommendations/42')
        self.assertEqual(503, response.status_code)
        self.assertEqual('service_not_configured', response.get_json()['error'])

    def test_missing_or_wrong_key_is_rejected_before_database_work(self):
        with patch.object(recommendation_app, 'fetch_all') as fetch:
            self.assertEqual(401, self.client.get('/recommendations/42').status_code)
            self.assertEqual(
                401,
                self.client.get(
                    '/recommendations/42',
                    headers={'X-Recommendation-Key': 'wrong-key'},
                ).status_code,
            )
            fetch.assert_not_called()

    def test_invalid_user_id_is_rejected(self):
        response = self.client.get(
            '/recommendations/0',
            headers={'X-Recommendation-Key': 'shared-test-key'},
        )
        self.assertEqual(422, response.status_code)
        self.assertEqual('invalid_user', response.get_json()['error'])

    def test_limit_is_integer_and_bounded_before_database_work(self):
        headers = {'X-Recommendation-Key': 'shared-test-key'}
        with patch.object(recommendation_app, 'fetch_all') as fetch:
            for value in ('invalid', '0', '21', '-1'):
                response = self.client.get(
                    f'/recommendations/42?limit={value}',
                    headers=headers,
                )
                self.assertEqual(422, response.status_code)
                self.assertEqual('invalid_limit', response.get_json()['error'])
            fetch.assert_not_called()

    def test_authenticated_request_reaches_recommendation_logic(self):
        with patch.object(recommendation_app, 'fetch_all', return_value=[]) as fetch:
            response = self.client.get(
                '/recommendations/42?limit=8',
                headers={'X-Recommendation-Key': 'shared-test-key'},
            )
            self.assertEqual(200, response.status_code)
            self.assertEqual([], response.get_json()['product_ids'])
            fetch.assert_called_once()


if __name__ == '__main__':
    unittest.main()