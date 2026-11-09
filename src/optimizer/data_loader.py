
from pathlib import Path

import pandas as pd


def load_inputs( predictions_path: str | Path, prices_path: str | Path ) -> tuple[pd.DataFrame, pd.DataFrame]:

    """
    Load the datasets. 

    Parameters
    ----------
    predictions_path : str | Path
        Path to  pred_oos.parquet file.

    prices_path : str | Path
        Path to prices.parquet file.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        predictions: Stock predictions. 
        prices: Historical stock prices.
    """

    predictions_file = Path(predictions_path)
    prices_file = Path(prices_path)

    # Check that both files exist.
    if not predictions_file.is_file():
        raise FileNotFoundError(
            f"Predictions file not found: {predictions_file}"
        )

    if not prices_file.is_file():
        raise FileNotFoundError(
            f"Prices file not found: {prices_file}"
        )

    # Load predictions.
    try:
        predictions = pd.read_parquet(
            predictions_file,
            engine="pyarrow"
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not read predictions file: {predictions_file}"
        ) from exc

    # Load historical prices.
    try:
        prices = pd.read_parquet(
            prices_file,
            engine="pyarrow"
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not read prices file: {prices_file}"
        ) from exc

    return predictions, prices

