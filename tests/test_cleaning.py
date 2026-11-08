import pandas as pd
from src.cleaning import clean_prices
from src.cleaning import clean_market
from pathlib import Path
from src.cleaning import quarantine_prices

# Test 1 : Normalisation des données
def test_clean_prices():

    data = {
        "Date": ["2026-11-05", "2026-11-04"],
        "Ticker": [" aapl ", "AAPL"],
        "Open": ["270", "268"],
        "High": ["275", "272"],
        "Low": ["268", "265"],
        "Close": ["273", "270"],
        "Volume": ["40000000", "35000000"]
    }

    df = pd.DataFrame(data)

    result = clean_prices(df)

    # Vérifier les colonnes
    assert "date" in result.columns
    assert "ticker" in result.columns

    # Vérifier les tickers
    assert result["ticker"].tolist() == ["AAPL", "AAPL"]

    # Vérifier les types
    assert pd.api.types.is_datetime64_any_dtype(result["date"])
    assert pd.api.types.is_numeric_dtype(result["close"])

    # Vérifier le tri chronologique par ticker
    assert result.groupby("ticker")["date"].apply(
        lambda dates: dates.is_monotonic_increasing
    ).all()


# Test 2 : Suppression des doublons
def test_duplicate_prices():

    data = {
        "date": ["2026-11-05", "2026-11-05"],
        "ticker": ["AAPL", "AAPL"],
        "open": [270, 270],
        "high": [275, 275],
        "low": [268, 268],
        "close": [273, 273],
        "volume": [40000000, 40000000]
    }

    df = pd.DataFrame(data)

    result = clean_prices(df)

    # Les deux lignes identiques deviennent une seule
    assert len(result) == 1

# Test 4 : Nettoyage des données de marché
def test_clean_market():

    data = {
        "Date": ["2026-11-06", "2026-11-05"],
        "SP500": ["6800.5", "6780.2"],
        "VIX": ["18.4", "19.2"]
    }

    df = pd.DataFrame(data)

    result = clean_market(df)

    # Vérifier les colonnes
    assert list(result.columns) == ["date", "sp500", "vix"]

    # Vérifier les types
    assert pd.api.types.is_datetime64_any_dtype(result["date"])
    assert pd.api.types.is_numeric_dtype(result["sp500"])
    assert pd.api.types.is_numeric_dtype(result["vix"])

    # Vérifier le tri chronologique
    assert result["date"].is_monotonic_increasing

    # Vérifier le nombre de lignes
    assert len(result) == 2


# Test 5 : Doublons des données de marché
def test_duplicate_market():

    data = {
        "date": ["2026-11-05", "2026-11-05"],
        "sp500": [6780.2, 6780.2],
        "vix": [19.2, 19.2]
    }

    df = pd.DataFrame(data)

    result = clean_market(df)

    # Deux lignes identiques deviennent une seule
    assert len(result) == 1

def test_clean_real_data():

    # Localiser les lots bruts
    prices_files = sorted(
        Path("data/raw/prices").glob("*.parquet")
    )

    market_files = sorted(
        Path("data/raw/market").glob("*.parquet")
    )

    # Ignorer ce test si les données locales sont absentes
    if not prices_files or not market_files:
        import pytest
        pytest.skip("Données brutes indisponibles")

    # Lire les fichiers réels
    prices = pd.concat(
        [pd.read_parquet(f) for f in prices_files],
        ignore_index=True
    )

    market = pd.concat(
        [pd.read_parquet(f) for f in market_files],
        ignore_index=True
    )

    # Nettoyer les données
    clean_p = clean_prices(prices)
    clean_m = clean_market(market)

    # Vérifier les colonnes
    assert set(
        ["date", "ticker", "open", "high", "low", "close", "volume"]
    ).issubset(clean_p.columns)

    assert set(
        ["date", "sp500", "vix"]
    ).issubset(clean_m.columns)

    # Vérifier les types des dates
    assert pd.api.types.is_datetime64_any_dtype(clean_p["date"])
    assert pd.api.types.is_datetime64_any_dtype(clean_m["date"])

    # Vérifier que les résultats ne sont pas vides
    assert not clean_p.empty
    assert not clean_m.empty

def test_quarantine_prices():

    df = pd.DataFrame({
        "date": pd.to_datetime([
            "2016-07-01",
            "2016-07-05",
            "2016-07-06",
            "2019-10-07",
            "2019-10-09",
            "2024-11-07"
        ]),
        "ticker": [
            "DHR", "DHR", "DHR",
            "IBKR", "IBKR", "APP"
        ]
    })

    accepted, rejected = quarantine_prices(df)

    assert len(accepted) == 3
    assert len(rejected) == 3

    assert "APP" in accepted["ticker"].values
    assert not (
        (accepted["ticker"] == "DHR")
        & (accepted["date"] <= pd.Timestamp("2016-07-05"))
    ).any()