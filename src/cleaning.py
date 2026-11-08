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

    # Retirer les séances sans données de marché complètes
    df = df.dropna(subset=["date", "sp500", "vix"])

    # 4. Supprimer les doublons strictement identiques
    df = df.drop_duplicates()

    # 5. Trier chronologiquement
    df = df.sort_values("date").reset_index(drop=True)

    return df

def quarantine_prices(df):
    df = df.copy()

    # DHR : période autour de la rupture historique
    dhr = (
        (df["ticker"] == "DHR")
        & (df["date"] <= pd.Timestamp("2016-07-05"))
    )

    # IBKR : deux séances suspectes
    ibkr = (
        (df["ticker"] == "IBKR")
        & (df["date"].isin([
            pd.Timestamp("2019-10-07"),
            pd.Timestamp("2019-10-08")
        ]))
    )

    rejected = df[dhr | ibkr].copy()
    accepted = df[~(dhr | ibkr)].copy()

    return accepted, rejected