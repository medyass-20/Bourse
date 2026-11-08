import pandas as pd


def check_prices(df):

    issues = {}

    # 1. Valeurs manquantes
    issues["missing_values"] = int(
        df.isna().any(axis=1).sum()
    )

    # 2. Doublons sur (ticker, date)
    issues["duplicate_keys"] = int(
        df.duplicated(["ticker", "date"]).sum()
    )

    # 3. Prix nuls ou négatifs
    prices = ["open", "high", "low", "close"]

    issues["invalid_prices"] = int(
        (df[prices] <= 0).any(axis=1).sum()
    )

    # 4. Incohérences OHLC
     # Vérification OHLC avec tolérance numérique
    tolerance = 1e-8

    issues["invalid_ohlc"] = int(
    (
        (df["high"] + tolerance < df[prices].max(axis=1))
        |
        (df["low"] - tolerance > df[prices].min(axis=1))
    ).sum()
   )
    # 5. Volumes négatifs
    issues["negative_volume"] = int(
        (df["volume"] < 0).sum()
    )

    return issues

def check_abnormal_returns(df, threshold=0.40):
    df = df.sort_values(["ticker", "date"]).copy()

    df["daily_return"] = (
        df.groupby("ticker")["close"]
        .pct_change(fill_method=None)
    )

    return df[df["daily_return"].abs() > threshold].copy()


def check_suspicious_volume(df):
    return df[
        (df["volume"] == 0)
        & (df["open"] == df["high"])
        & (df["high"] == df["low"])
        & (df["low"] == df["close"])
    ].copy()    

def quality_report(df):

    returns = check_abnormal_returns(df).copy()
    returns["anomaly"] = "abnormal_return"

    volumes = check_suspicious_volume(df).copy()
    volumes["anomaly"] = "suspicious_volume"

    report = pd.concat(
        [returns, volumes],
        ignore_index=True
    )

    return report.sort_values(
        ["date", "ticker"]
    ).reset_index(drop=True)