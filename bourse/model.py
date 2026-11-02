from __future__ import annotations


class LinearRegressionGD:
    def __init__(self, learning_rate: float = 1e-7, epochs: int = 5000) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if epochs <= 0:
            raise ValueError("epochs must be > 0")
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights: list[float] = []
        self.bias = 0.0

    def fit(self, x: list[list[float]], y: list[float]) -> None:
        if not x or not y:
            raise ValueError("x and y must not be empty")
        if len(x) != len(y):
            raise ValueError("x and y must have same length")

        n_samples = len(x)
        n_features = len(x[0])
        if n_features == 0:
            raise ValueError("x must contain at least one feature")
        if any(len(row) != n_features for row in x):
            raise ValueError("all rows in x must have same feature count")

        self.weights = [0.0 for _ in range(n_features)]
        self.bias = 0.0

        for _ in range(self.epochs):
            dw = [0.0 for _ in range(n_features)]
            db = 0.0

            for i in range(n_samples):
                prediction = self._predict_row(x[i])
                error = prediction - y[i]
                for j in range(n_features):
                    dw[j] += error * x[i][j]
                db += error

            for j in range(n_features):
                self.weights[j] -= self.learning_rate * (dw[j] / n_samples)
            self.bias -= self.learning_rate * (db / n_samples)

    def predict(self, x: list[list[float]]) -> list[float]:
        if not self.weights:
            raise ValueError("model has not been fitted")
        if not x:
            return []
        if any(len(row) != len(self.weights) for row in x):
            raise ValueError("feature size mismatch")
        return [self._predict_row(row) for row in x]

    def _predict_row(self, row: list[float]) -> float:
        return sum(weight * value for weight, value in zip(self.weights, row)) + self.bias

