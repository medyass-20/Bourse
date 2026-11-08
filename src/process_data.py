from pathlib import Path
import pandas as pd

from src.cleaning import clean_prices, clean_market, quarantine_prices
from src.quality_check import check_prices, quality_report


ROOT = Path(__file__).resolve().parents[1]


def main(root=ROOT):

    # Lire les données brutes
    prices = pd.read_parquet(
        root / "data/raw/prices/prices.parquet"
        )
    market = pd.read_parquet(
        root  / "data/raw/market/market.parquet"
        )

    # Nettoyer
    prices = clean_prices(prices)
    market = clean_market(market)

    # Contrôler
    issues = check_prices(prices)

    if any(issues.values()):
        raise ValueError(f"Anomalies structurelles : {issues}")

    # Détecter les anomalies financières
    anomalies = quality_report(prices)

    # Appliquer la quarantaine
    accepted, rejected = quarantine_prices(prices)

    # Préparer les dossiers
    (root / "data/processed").mkdir(parents=True, exist_ok=True)
    (root / "reports").mkdir(parents=True, exist_ok=True)

    # Enregistrer les résultats
    accepted.to_parquet(
        root / "data/processed/prices.parquet",
        index=False
    )

    market.to_parquet(
        root / "data/processed/market.parquet",
        index=False
    )

    rejected.to_parquet(
        root / "reports/quarantine_prices.parquet",
        index=False
    )

    anomalies.to_csv(
        root / "reports/quality_anomalies.csv",
        index=False
    )

    print("Données traitées :", len(accepted))
    print("Données isolées :", len(rejected))
    print("Alertes financières :", len(anomalies))


if __name__ == "__main__":
    main()