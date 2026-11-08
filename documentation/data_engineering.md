# Bourse X — Data Engineering

## 1. Objectif

Le pipeline Data Engineering prépare les données historiques des actions américaines ainsi que celles du S&P 500 et du VIX pour les modules de Machine Learning, d'optimisation et de backtesting.

## 2. Source et organisation

Les données proviennent de Yahoo Finance via `yfinance`.

- `tickers.txt` : univers de 100 symboles.
- `src/ingestion.py` : téléchargement et mise à jour des données.
- `src/cleaning.py` : normalisation et traitement des observations.
- `src/quality_check.py` : détection des anomalies.
- `src/process_data.py` : orchestration et export.

Les fichiers bruts sont conservés dans `data/raw/` et les fichiers traités sont produits dans `data/processed/`.

Les données téléchargées sont exclues du suivi Git. Leur utilisation et leur partage doivent respecter les conditions du fournisseur.

## 3. Contrôles qualité

Les contrôles implémentés couvrent :

- Les valeurs manquantes.
- Les doublons.
- Les prix nuls ou négatifs.
- Les incohérences OHLC, avec une tolérance numérique.
- Les volumes négatifs.
- Les variations de clôture supérieures à 40 %.
- Les séances à volume nul avec OHLC identiques.

Un rendement exceptionnel déclenche une alerte, sans entraîner automatiquement une exclusion.

## 4. Anomalies historiques et décisions provisoires

| Ticker | Date | Observation | Traitement |
|---|---|---|---|
| AMD | 2016-04-22 | Rendement +52,29 % | Conservé |
| APP | 2024-11-07 | Rendement +46,27 % | Conservé provisoirement |
| DHR | 2016-07-05 | Rupture de prix +61,22 % | Historique jusqu'au 05/07/2016 isolé provisoirement |
| IBKR | 2019-10-07 et 2019-10-08 | Volume nul, OHLC identiques | Deux observations isolées provisoirement |

Les 33 alertes OHLC initiales provenaient d'écarts de précision numérique de l'ordre de 10⁻¹⁴ et sont désormais ignorées grâce à une tolérance.

Les exclusions DHR et IBKR sont des mesures conservatoires pour le MVP. Elles ne constituent pas une correction définitive des séries historiques.

## 5. Résultats du traitement

Dernière exécution vérifiée :

- 269 725 observations d'actions en entrée.
- 269 560 observations conservées.
- 165 observations isolées.
- 5 alertes financières détectées.

Deux dates sans clôture S&P 500, le 25 mai et le 7 septembre 2026, sont retirées du fichier de marché traité.

Les fichiers `data/raw/` ne sont jamais modifiés par le pipeline de nettoyage.

## 6. Exécution

Installer les dépendances du projet, puis exécuter depuis la racine :

```bash
python -m src.ingestion
python -m src.process_data
python -m pytest tests/ -v
```

Pour reconstruire tout l'historique après un changement de l'univers :

```bash
python -m src.ingestion --full-refresh
python -m src.process_data
```

Les données doivent être téléchargées et vérifiées localement avant le traitement.

## 7. Limites et précautions

- L'univers actuel de 100 actions n'est pas une reconstitution historique de la composition du S&P 500. Les backtests doivent tenir compte du biais de survivance et des informations utilisées pour constituer cet univers.
- Les données DHR antérieures au 6 juillet 2016 restent exclues dans la version actuelle, en attente d'une correction fiable.
- Les deux séances IBKR isolées créent une interruption dans l'historique ; les modules de features et de labels doivent en tenir compte.
- Les données ne doivent pas être considérées comme exhaustivement validées avant les derniers tests d'intégration.
- Les résultats du modèle sont expérimentaux et ne constituent pas des recommandations d'investissement.

## Précautions pour le Machine Learning

### Interruptions des historiques

Certaines observations ont été mises en quarantaine.

Exemple : IBKR, les 7 et 8 octobre 2019.

Après exclusion, les séances du 4 et du 9 octobre deviennent
adjacentes dans le fichier traité, mais elles ne sont pas
consécutives dans le calendrier boursier.

Le module features.py doit :
- Utiliser un calendrier de séances de référence.
- Détecter les interruptions par ticker.
- Éviter de calculer des rendements et labels à travers ces interruptions.

### DHR

Les observations DHR jusqu'au 5 juillet 2016 sont exclues
provisoirement à cause d'une rupture historique suspecte.

Cette exclusion réduit l'historique disponible.
Elle pourra être révisée après vérification des prix ajustés.

### Biais de survivance

L'univers actuel de 100 actions est basé sur une sélection
récente. Il ne représente pas nécessairement les entreprises
présentes dans le S&P 500 à chaque date historique.

Les résultats du backtest doivent mentionner ce biais.
