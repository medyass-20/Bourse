from __future__ import annotations

import random


def generate_synthetic_prices(
    days: int = 120,
    start_price: float = 100.0,
    drift: float = 0.0008,
    volatility: float = 0.015,
    seed: int = 42,
) -> list[float]:
    if days < 2:
        raise ValueError("days must be >= 2")
    if start_price <= 0:
        raise ValueError("start_price must be > 0")

    rng = random.Random(seed)
    prices = [start_price]
    for _ in range(days - 1):
        shock = rng.gauss(drift, volatility)
        next_price = max(0.01, prices[-1] * (1 + shock))
        prices.append(next_price)
    return prices


def create_feature_matrix(prices: list[float], window: int = 5) -> tuple[list[list[float]], list[float]]:
    if window < 2:
        raise ValueError("window must be >= 2")
    if len(prices) <= window:
        raise ValueError("not enough prices for selected window")

    features: list[list[float]] = []
    targets: list[float] = []

    for i in range(window, len(prices)):
        history = prices[i - window : i]
        mean_price = sum(history) / window
        momentum = history[-1] - history[0]
        last_price = history[-1]

        features.append([mean_price, momentum, last_price])
        targets.append(prices[i])

    return features, targets

