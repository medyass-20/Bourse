import pandas as pd
from src.cleaning import clean_prices
from src.cleaning import clean_prices, clean_market


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
