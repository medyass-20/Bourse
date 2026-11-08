import pandas as pd
from src.quality_check import (
    check_prices,
    check_abnormal_returns,
    check_suspicious_volume
)


def test_check_prices():

    df = pd.DataFrame({
        "date": ["2026-11-05", "2026-11-06"],
        "ticker": ["AAPL", "AAPL"],
        "open": [270, -10],
        "high": [275, 12],
        "low": [268, 15],
        "close": [273, 11],
        "volume": [40000000, 1000000]
    })

    result = check_prices(df)

    assert result["invalid_prices"] == 1
    assert result["invalid_ohlc"] == 1
    assert result["missing_values"] == 0
    assert result["duplicate_keys"] == 0
    assert result["negative_volume"] == 0

def test_abnormal_returns():
    df = pd.DataFrame({
        "date": ["2016-07-01", "2016-07-05"],
        "ticker": ["DHR", "DHR"],
        "close": [41.9, 67.6]
    })

    result = check_abnormal_returns(df)

    assert len(result) == 1
    assert result.iloc[0]["ticker"] == "DHR"


def test_suspicious_volume():
    df = pd.DataFrame({
        "date": ["2019-10-07", "2019-10-08"],
        "ticker": ["IBKR", "IBKR"],
        "open": [45, 45],
        "high": [45, 45],
        "low": [45, 45],
        "close": [45, 45],
        "volume": [0, 0]
    })

    result = check_suspicious_volume(df)

    assert len(result) == 2
    assert (result["ticker"] == "IBKR").all()