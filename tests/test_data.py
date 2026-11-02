import unittest

from bourse.data import create_feature_matrix, generate_synthetic_prices


class DataTests(unittest.TestCase):
    def test_generate_synthetic_prices_has_expected_length(self) -> None:
        prices = generate_synthetic_prices(days=30, seed=1)
        self.assertEqual(len(prices), 30)
        self.assertTrue(all(price > 0 for price in prices))

    def test_create_feature_matrix_shapes(self) -> None:
        prices = [10, 11, 12, 13, 14, 15]
        x, y = create_feature_matrix(prices, window=3)
        self.assertEqual(len(x), 3)
        self.assertEqual(len(y), 3)
        self.assertEqual(len(x[0]), 3)


if __name__ == "__main__":
    unittest.main()

