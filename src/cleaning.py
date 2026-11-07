import pandas as pd


def clean_prices(df):
    df = df.copy()

    # Normaliser les noms des colonnes
    df.columns = df.columns.str.strip().str.lower()

    # Convertir les dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Convertir les colonnes numériques
    numeric_cols = ["open", "high", "low", "close", "volume"]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normaliser les tickers
    df["ticker"] = df["ticker"].astype("string").str.strip().str.upper()

    # Trier les données
    df = df.sort_values(["ticker", "date"])

    #supprimer les doublons
    df = df.drop_duplicates()

    #trier les colonnes
    df = df.sort_values(["ticker", "date"])

    return df

def clean_market(df):
    df = df.copy()

    # 1. Normaliser les noms des colonnes
    df.columns = df.columns.str.strip().str.lower()

    # 2. Convertir les dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 3. Convertir les valeurs numériques
    numeric_cols = ["sp500", "vix"]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Supprimer les doublons strictement identiques
    df = df.drop_duplicates()

    # 5. Trier chronologiquement
    df = df.sort_values("date").reset_index(drop=True)

    return df