import pandas as pd
from src.ingestion import load_tickers
from src.process_data import main


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

def test_pipeline_reproducibility(tmp_path):

    # Préparer les dossiers temporaires
    raw_prices = tmp_path / "data/raw/prices"
    raw_market = tmp_path / "data/raw/market"

    raw_prices.mkdir(parents=True)
    raw_market.mkdir(parents=True)

    # Simuler des données d'actions
    prices = pd.DataFrame({
        "date": pd.to_datetime([
            "2016-07-01",
            "2016-07-05",
            "2016-07-06",
            "2019-10-07",
            "2019-10-09"
        ]),
        "ticker": ["DHR", "DHR", "DHR", "IBKR", "IBKR"],
        "open": [41, 65, 67, 11, 11],
        "high": [42, 68, 69, 11, 12],
        "low": [40, 64, 66, 11, 10],
        "close": [41, 67, 68, 11, 11],
        "volume": [1000, 2000, 3000, 0, 2500]
    })

    prices.to_parquet(raw_prices / "prices.parquet", index=False)

    # Simuler les données de marché
    market = pd.DataFrame({
        "date": pd.to_datetime([
            "2016-07-05",
            "2016-07-06",
            "2019-10-07"
        ]),
        "sp500": [2100, 2110, None],
        "vix": [18, 17, 20]
    })

    market.to_parquet(raw_market / "market.parquet", index=False)

    # Exécuter le pipeline dans le dossier temporaire
    main(root=tmp_path)

    # Vérifier les fichiers produits
    processed_prices = pd.read_parquet(
        tmp_path / "data/processed/prices.parquet"
    )

    processed_market = pd.read_parquet(
        tmp_path / "data/processed/market.parquet"
    )

    quarantine = pd.read_parquet(
        tmp_path / "reports/quarantine_prices.parquet"
    )

    # DHR 01/07 et 05/07 + IBKR 07/10 isolés
    assert len(quarantine) == 3

    # DHR 06/07 et IBKR 09/10 conservés
    assert len(processed_prices) == 2

    # La ligne de marché avec sp500 manquant est retirée
    assert len(processed_market) == 2

    # Les données brutes sont toujours intactes
    assert len(pd.read_parquet(raw_prices / "prices.parquet")) == 5

def test_ibkr_gap():

    prices = pd.read_parquet(
        "data/processed/prices.parquet"
    )

    ibkr = prices[
        (prices["ticker"] == "IBKR")
        & (prices["date"].between("2019-10-03", "2019-10-11"))
    ].sort_values("date")

    dates = ibkr["date"].dt.strftime("%Y-%m-%d").tolist()

    assert "2019-10-04" in dates
    assert "2019-10-07" not in dates
    assert "2019-10-08" not in dates
    assert "2019-10-09" in dates