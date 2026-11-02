from __future__ import annotations

import argparse

from .data import create_feature_matrix, generate_synthetic_prices
from .model import LinearRegressionGD


def run_prediction(days: int, window: int, seed: int) -> float:
    prices = generate_synthetic_prices(days=days, seed=seed)
    x, y = create_feature_matrix(prices, window=window)

    model = LinearRegressionGD()
    model.fit(x, y)

    latest_features = [x[-1]]
    next_price = model.predict(latest_features)[0]
    return next_price


def main() -> None:
    parser = argparse.ArgumentParser(description="Prédiction boursière simple")
    parser.add_argument("--days", type=int, default=120, help="Nombre de jours simulés")
    parser.add_argument("--window", type=int, default=5, help="Fenêtre historique utilisée")
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire")
    args = parser.parse_args()

    prediction = run_prediction(days=args.days, window=args.window, seed=args.seed)
    print(f"Prix prédit pour le prochain jour: {prediction:.2f}")


if __name__ == "__main__":
    main()

