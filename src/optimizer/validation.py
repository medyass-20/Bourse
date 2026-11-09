import numpy as np
import pandas as pd


def validate_input_data( predictions: pd.DataFrame, prices: pd.DataFrame ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate and normalize predictions and prices.

    Returns validated copies of both DataFrames.
    Raises ValueError if an essential validation fails.
    """

    predictions = predictions.copy()
    prices = prices.copy()

    # --------------------------------------------------
    # 1. Check required columns
    # --------------------------------------------------

    required_prediction_columns = {
        "date", "ticker", "score", "rank"
    }

    required_price_columns = {
        "date", "ticker", "close"
    }

    missing_prediction_columns = (
        required_prediction_columns - set(predictions.columns)
    )

    missing_price_columns = (
        required_price_columns - set(prices.columns)
    )

    if missing_prediction_columns:
        raise ValueError(
            f"Missing prediction columns: "
            f"{missing_prediction_columns}"
        )

    if missing_price_columns:
        raise ValueError(
            f"Missing price columns: {missing_price_columns}"
        )

    # --------------------------------------------------
    # 2. Normalize dates
    # --------------------------------------------------

    predictions["date"] = pd.to_datetime(
        predictions["date"], errors="coerce"
    )

    prices["date"] = pd.to_datetime(
        prices["date"], errors="coerce"
    )

    if predictions["date"].isna().any():
        raise ValueError("Predictions contain invalid dates.")

    if prices["date"].isna().any():
        raise ValueError("Prices contain invalid dates.")

    # --------------------------------------------------
    # 3. Normalize tickers
    # --------------------------------------------------

    for df, name in [
        (predictions, "Predictions"),
        (prices, "Prices")
    ]:
        df["ticker"] = df["ticker"].astype("string").str.strip().str.upper()

        if df["ticker"].isna().any() or df["ticker"].eq("").any():
            raise ValueError(
                f"{name} contain missing or empty tickers."
            )

    # --------------------------------------------------
    # 4. Check missing values
    # --------------------------------------------------

    prediction_columns = [
        "date", "ticker", "score", "rank"
    ]

    price_columns = [
        "date", "ticker", "close"
    ]

    if predictions[prediction_columns].isna().any().any():
        raise ValueError(
            "Predictions contain missing required values."
        )

    if prices[price_columns].isna().any().any():
        raise ValueError(
            "Prices contain missing required values."
        )

    # --------------------------------------------------
    # 5. Validate prediction scores
    # --------------------------------------------------

    predictions["score"] = pd.to_numeric(
        predictions["score"], errors="coerce"
    )

    if (
        predictions["score"].isna().any()
        or not np.isfinite(predictions["score"]).all()
    ):
        raise ValueError("Prediction scores must be finite numbers.")

    # --------------------------------------------------
    # 6. Validate prediction ranks
    # --------------------------------------------------

    predictions["rank"] = pd.to_numeric(
        predictions["rank"], errors="coerce"
    )

    if (
        predictions["rank"].isna().any()
        or not np.isfinite(predictions["rank"]).all()
        or (predictions["rank"] < 1).any()
        or (predictions["rank"] % 1 != 0).any()
    ):
        raise ValueError(
            "Ranks must be positive integers."
        )

    predictions["rank"] = predictions["rank"].astype(int)

    # --------------------------------------------------
    # 7. Validate closing prices
    # --------------------------------------------------

    prices["close"] = pd.to_numeric(
        prices["close"], errors="coerce"
    )

    if (
        prices["close"].isna().any()
        or not np.isfinite(prices["close"]).all()
        or (prices["close"] <= 0).any()
    ):
        raise ValueError(
            "Closing prices must be finite positive numbers."
        )

    # --------------------------------------------------
    # 8. Check duplicate records
    # --------------------------------------------------

    if predictions.duplicated(
        subset=["date", "ticker"]
    ).any():
        raise ValueError(
            "Duplicate (date, ticker) records in predictions."
        )

    if prices.duplicated(
        subset=["date", "ticker"]
    ).any():
        raise ValueError(
            "Duplicate (date, ticker) records in prices."
        )

    # --------------------------------------------------
    # 9. Check ticker availability
    # --------------------------------------------------

    predicted_tickers = set(predictions["ticker"])
    available_tickers = set(prices["ticker"])

    missing_tickers = predicted_tickers - available_tickers

    if missing_tickers:
        raise ValueError(
            f"No historical prices found for: {missing_tickers}"
        )

    return predictions, prices

