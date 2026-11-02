import unittest

from bourse.model import LinearRegressionGD
from bourse.predict import run_prediction


class ModelTests(unittest.TestCase):
    def test_linear_regression_learns_simple_relation(self) -> None:
        x = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        y = [5.0, 7.0, 9.0, 11.0, 13.0]  # y = 2x + 3

        model = LinearRegressionGD(learning_rate=0.01, epochs=5000)
        model.fit(x, y)
        prediction = model.predict([[6.0]])[0]

        self.assertAlmostEqual(prediction, 15.0, delta=0.2)

    def test_end_to_end_prediction_returns_float(self) -> None:
        prediction = run_prediction(days=120, window=5, seed=42)
        self.assertIsInstance(prediction, float)
        self.assertGreater(prediction, 0.0)


if __name__ == "__main__":
    unittest.main()

