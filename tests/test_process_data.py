import pandas as pd
from src.ingestion import load_tickers


def test_processed_data():

    prices = pd.read_parquet(
        "data/processed/prices.parquet"
    )

    market = pd.read_parquet(
        "data/processed/market.parquet"
    )

    # Les données existent
    assert not prices.empty
    assert not market.empty

    # Aucune clé dupliquée
    assert not prices.duplicated(["ticker", "date"]).any()
    assert not market.duplicated("date").any()

    # Vérifier les exclusions de DHR
    dhr = prices[prices["ticker"] == "DHR"]

    assert (
        dhr["date"].min() > pd.Timestamp("2016-07-05")
    )

    # Vérifier les exclusions de IBKR
    ibkr = prices[prices["ticker"] == "IBKR"]

    assert not ibkr["date"].isin([
        pd.Timestamp("2019-10-07"),
        pd.Timestamp("2019-10-08")
    ]).any()

def test_processed_market():

    market = pd.read_parquet(
        "data/processed/market.parquet"
    )

    assert list(market.columns) == [
        "date", "sp500", "vix"
    ]

    print("\nValeurs manquantes par colonne :")
    print(market.isna().sum())

    print("\nLignes avec valeurs manquantes :")
    print(market[market.isna().any(axis=1)].to_string(index=False))

    assert not market.isna().any().any()

    assert (market["sp500"] > 0).all()
    assert (market["vix"] > 0).all()

    assert not market["date"].duplicated().any()
    assert market["date"].is_monotonic_increasing

def test_processed_tickers():

    prices = pd.read_parquet(
        "data/processed/prices.parquet"
    )

    expected = set(load_tickers())
    actual = set(prices["ticker"].unique())

    assert actual == expected