from pathlib import Path

from src.quality_check import quality_report

import pandas as pd
import pytest

from src.cleaning import clean_prices
from src.quality_check import (
    check_prices,
    check_abnormal_returns,
    check_suspicious_volume
)


def test_quality_real_data():

    path = Path("data/raw/prices/prices.parquet")

    if not path.exists():
        pytest.skip("Fichier Parquet introuvable")

    # Charger et nettoyer les données
    df = clean_prices(pd.read_parquet(path))

    # Contrôles structurels
    issues = check_prices(df)
    print("\nContrôles structurels :", issues)

        # Identifier les incohérences OHLC
    prices = ["open", "high", "low", "close"]

    invalid = df[
        (df["high"] < df[prices].max(axis=1))
        |
        (df["low"] > df[prices].min(axis=1))
    ]

    print("\nIncohérences OHLC :")
    print(
        invalid[
            ["date", "ticker", "open", "high", "low", "close"]
        ].to_string(index=False)
    )

    # Détecter les rendements suspects
    returns = check_abnormal_returns(df)
    print("\nVariations anormales :")
    print(returns[["date", "ticker", "daily_return"]].to_string(index=False))

    # Détecter les volumes suspects
    volumes = check_suspicious_volume(df)
    print("\nVolumes suspects :")
    print(volumes[["date", "ticker", "volume"]].to_string(index=False))

        # Mesurer les écarts OHLC exacts
    diff_high = df[["open", "close", "low"]].max(axis=1) - df["high"]
    diff_low = df["low"] - df[["open", "close", "high"]].min(axis=1)

    print("\nÉcart HIGH maximal :", diff_high.max())
    print("Écart LOW maximal :", diff_low.max())

    # Vérifier que le fichier contient des données
    assert not df.empty

    report = quality_report(df)

    print("\nRapport des anomalies :")
    print(
        report[
            ["date", "ticker", "anomaly"]
        ].to_string(index=False)
    )

    assert len(report) == 5

        # Examiner les observations autour des anomalies
    cases = [
        ("DHR", "2016-06-28", "2016-07-12"),
        ("IBKR", "2019-10-03", "2019-10-11"),
        ("AMD", "2016-04-19", "2016-04-26"),
        ("APP", "2024-11-04", "2024-11-12"),
    ]

    for ticker, start, end in cases:
        subset = df[
            (df["ticker"] == ticker)
            & (df["date"] >= pd.Timestamp(start))
            & (df["date"] <= pd.Timestamp(end))
        ]

        print(f"\nHistorique autour de {ticker} :")
        print(
            subset[
                ["date", "open", "high", "low", "close", "volume"]
            ].to_string(index=False)
        )