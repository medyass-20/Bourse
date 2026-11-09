import pandas as pd


def select_top_stocks( predictions: pd.DataFrame, top_n: int, prediction_date: str | pd.Timestamp | None = None ) -> pd.DataFrame:
    """
    Select the Top N stocks from predictions.

    Parameters
    ----------
    predictions : pd.DataFrame
        predictions containing date, ticker, score and rank.

    top_n : int
        Number of stocks to select.

    prediction_date : str | pd.Timestamp | None
        Date for which predictions should be selected.
        If None, predictions must contain only one date.

    Returns
    -------
    pd.DataFrame
        Selected stocks, ordered by rank.
    """

    # 1. Validate top_n
    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n <= 0:
        raise ValueError("top_n must be a positive integer.")

    # 2. Ensure only one prediction date is selected
    if prediction_date is not None:
        prediction_date = pd.to_datetime(
            prediction_date,
            errors="raise"
        )

        daily_predictions = predictions[
            predictions["date"] == prediction_date
        ].copy()

        if daily_predictions.empty:
            raise ValueError(
                f"No predictions found for {prediction_date.date()}."
            )

    else:
        unique_dates = predictions["date"].drop_duplicates()

        if len(unique_dates) != 1:
            raise ValueError(
                "Predictions contain multiple dates. "
                "Specify prediction_date explicitly."
            )

        daily_predictions = predictions.copy()

    # 3. Check available number of stocks
    if len(daily_predictions) < top_n:
        raise ValueError(
            f"Requested {top_n} stocks, but only "
            f"{len(daily_predictions)} are available."
        )

    # 4. Sort by rank, then score, then ticker
    daily_predictions = daily_predictions.sort_values(
        by=["rank", "score", "ticker"],
        ascending=[True, False, True]
    )

    # 5. Select Top N
    selected_stocks = daily_predictions.head(top_n).reset_index(
        drop=True
    )

    return selected_stocks

