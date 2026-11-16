# -*- coding: utf-8 -*-
"""Phase Features : construit le jeu de données d'entraînement (features + label).

Entrées  : data/processed/prices.parquet et market.parquet (sortie du nettoyage).
Sorties  : data/processed/dataset.parquet  (date, ticker, features..., excess_ret, label)
           data/processed/dataset_info.json (journal : périmètre, lignes retirées, hash des entrées)

Conforme au cahier des charges :
  - section 6  : familles momentum, tendance, volatilité, volume, oscillateurs,
                 structure de séance, relatif au marché, contexte externe ;
  - section 3.3: label = excès de rendement du lendemain vs médiane du jour,
                 30 % hauts -> 1, 30 % bas -> 0, 40 % du milieu exclus ;
  - F04        : aucune information postérieure à la clôture de t dans les features ;
  - traitement : normalisation transversale par date (rang centile), valeurs
                 manquantes des fenêtres incomplètes supprimées (documenté).

Pourquoi pas de winsorisation ni de z-score globaux : calculés sur tout
l'échantillon, ils injecteraient de l'information du futur dans le passé
(fuite). Le rang par date remplit le même rôle sans fuite.

Usage :
    python -m src.features                 # 13 features (core, défaut)
    python -m src.features --set full      # 24 features (CDC complet)
    python -m src.features --prices CHEMIN --market CHEMIN --out CHEMIN

Codes de sortie : 0 = ok, 2 = données refusées (validation), 1 = plantage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
PROC = BASE / "data" / "processed"
IN_PRICES = PROC / "prices.parquet"
IN_MARKET = PROC / "market.parquet"
OUT_DATASET = PROC / "dataset.parquet"
OUT_INFO = PROC / "dataset_info.json"

# Fenêtres (jours de bourse)
RSI_N = 14
MACD_FAST, MACD_SLOW, MACD_SIG = 12, 26, 9
BETA_N = 60
LABEL_Q_HI = 0.70   # 30 % les plus hauts -> label 1
LABEL_Q_LO = 0.30   # 30 % les plus bas  -> label 0

# Les 20 features propres à chaque action (transformées en rang par date, suffixe _rk)
TICKER_FEATURES = [
    # Momentum
    "ret_1", "ret_2", "ret_5", "ret_10", "ret_20",
    # Tendance
    "dist_mm5", "dist_mm20", "dist_mm50", "croise_mm5_mm20",
    # Volatilité
    "vol_5", "vol_20", "range_1", "atr_14",
    # Volume
    "volume_rel", "vol_chg_5",
    # Oscillateurs
    "rsi_14", "macd_hist",
    # Structure de la séance
    "gap_open", "close_pos",
    # Relatif au marché
    "beta_60",
    # NB : "rendement moins l'indice" (CDC) a été retiré : une fois transformé
    # en rang par date, il est identique à ret_20_rk (corrélation mesurée : 1.000),
    # car soustraire une constante du jour ne change pas le classement du jour.
]
# Les 4 features de contexte (identiques pour toutes les actions -> pas de rang)
MARKET_FEATURES = ["sp500_ret_1", "sp500_ret_5", "vix", "vix_chg_5"]
RANKED = [f + "_rk" for f in TICKER_FEATURES]
ALL_FEATURES = RANKED + MARKET_FEATURES

# Jeu "core" : 9 features par action + 4 de marché = 13 (un représentant par famille
# principale, volontairement court pour une première base qui tourne).
# Jeu "full" : les 20 + 4 = 24, conformes à toute la section 6 du CDC.
CORE_TICKER_FEATURES = [
    "ret_1", "ret_5", "ret_10", "ret_20",      # momentum
    "dist_mm20",                               # tendance / retour à la moyenne
    "vol_20", "range_1",                       # volatilité
    "volume_rel",                              # volume
    "beta_60",                                 # sensibilité au marché
]
FEATURE_SETS = {"core": CORE_TICKER_FEATURES, "full": TICKER_FEATURES}
DEFAULT_SET = "core"


def feature_columns(feature_set: str = DEFAULT_SET) -> list[str]:
    """Colonnes de features du dataset final pour un jeu donné (rangs + marché)."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"jeu inconnu {feature_set!r} (choix : {sorted(FEATURE_SETS)})")
    return [f + "_rk" for f in FEATURE_SETS[feature_set]] + MARKET_FEATURES


# ----------------------------------------------------------------------------- entrées
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_inputs(prices_path: Path, market_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge et REFUSE les données qui ne respectent pas le contrat."""
    problemes: list[str] = []
    for p in (prices_path, market_path):
        if not Path(p).exists():
            raise SystemExit(f"ERREUR : fichier manquant {p} — lancer d'abord le nettoyage.")
    prices = pd.read_parquet(prices_path)
    market = pd.read_parquet(market_path)

    attendu_p = {"date", "ticker", "open", "high", "low", "close", "volume"}
    attendu_m = {"date", "sp500", "vix"}
    if not attendu_p.issubset(prices.columns):
        problemes.append(f"prices : colonnes manquantes {attendu_p - set(prices.columns)}")
    if not attendu_m.issubset(market.columns):
        problemes.append(f"market : colonnes manquantes {attendu_m - set(market.columns)}")
    if problemes:
        raise ValueError("Contrat de données violé : " + " ; ".join(problemes))

    prices = prices.copy()
    market = market.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    market["date"] = pd.to_datetime(market["date"])
    prices["ticker"] = prices["ticker"].astype(str)

    if prices.duplicated(["date", "ticker"]).any():
        problemes.append("prices : doublons (date, ticker)")
    if market.duplicated(["date"]).any():
        problemes.append("market : dates en double")
    if prices["close"].isna().any():
        problemes.append("prices : close manquant (le nettoyage doit traiter cela)")
    if market[["sp500", "vix"]].isna().any().any():
        problemes.append("market : sp500 ou vix manquant")
    dates_hors_calendrier = set(prices["date"]) - set(market["date"])
    if dates_hors_calendrier:
        problemes.append(
            f"{len(dates_hors_calendrier)} dates de prices absentes de market "
            f"(ex. {sorted(dates_hors_calendrier)[:3]})"
        )
    if problemes:
        raise ValueError("Contrat de données violé : " + " ; ".join(problemes))

    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    market = market.sort_values("date").reset_index(drop=True)
    return prices, market


# ----------------------------------------------------------------------------- features
def market_features(market: pd.DataFrame) -> pd.DataFrame:
    """Features de contexte, par date."""
    m = market.sort_values("date").copy()
    m["sp500_ret_1"] = m["sp500"] / m["sp500"].shift(1) - 1
    m["sp500_ret_5"] = m["sp500"] / m["sp500"].shift(5) - 1
    m["vix_chg_5"] = m["vix"] - m["vix"].shift(5)
    return m[["date", "sp500_ret_1", "sp500_ret_5", "vix", "vix_chg_5"]]


def _rsi(close: pd.Series, n: int = RSI_N) -> pd.Series:
    """RSI de Wilder. L'ewm (adjust=False) n'utilise que le passé."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi.fillna(100.0).where(avg_gain.notna() & avg_loss.notna())


def _ticker_features(g: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Features d'UN ticker (g trié par date, avec colonnes marché déjà jointes)."""
    out = g[["date"]].copy()
    out["ticker"] = ticker
    c, v = g["close"], g["volume"]
    r = c.pct_change()

    # Momentum
    for k in (1, 2, 5, 10, 20):
        out[f"ret_{k}"] = c / c.shift(k) - 1
    # Tendance
    mm5 = c.rolling(5, min_periods=5).mean()
    mm20 = c.rolling(20, min_periods=20).mean()
    mm50 = c.rolling(50, min_periods=50).mean()
    out["dist_mm5"] = c / mm5 - 1
    out["dist_mm20"] = c / mm20 - 1
    out["dist_mm50"] = c / mm50 - 1
    out["croise_mm5_mm20"] = mm5 / mm20 - 1
    # Volatilité
    out["vol_5"] = r.rolling(5, min_periods=5).std()
    out["vol_20"] = r.rolling(20, min_periods=20).std()
    out["range_1"] = (g["high"] - g["low"]) / c
    # ATR 14 (Wilder), normalisé par le prix pour être comparable entre actions
    prev_c = c.shift(1)
    tr = pd.concat([
        g["high"] - g["low"],
        (g["high"] - prev_c).abs(),
        (g["low"] - prev_c).abs(),
    ], axis=1).max(axis=1)
    out["atr_14"] = tr.ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean() / c
    # Volume
    vol_mm5 = v.rolling(5, min_periods=5).mean()
    vol_mm20 = v.rolling(20, min_periods=20).mean()
    out["volume_rel"] = v / vol_mm20.replace(0.0, np.nan)
    out["vol_chg_5"] = vol_mm5 / vol_mm20.replace(0.0, np.nan) - 1
    # Oscillateurs
    out["rsi_14"] = _rsi(c)
    ema_f = c.ewm(span=MACD_FAST, adjust=False, min_periods=MACD_FAST).mean()
    ema_s = c.ewm(span=MACD_SLOW, adjust=False, min_periods=MACD_SLOW).mean()
    macd = ema_f - ema_s
    signal = macd.ewm(span=MACD_SIG, adjust=False, min_periods=MACD_SIG).mean()
    out["macd_hist"] = (macd - signal) / c  # normalisé par le prix pour être comparable
    # Structure de la séance
    out["gap_open"] = g["open"] / c.shift(1) - 1
    amplitude = (g["high"] - g["low"])
    out["close_pos"] = np.where(
        amplitude > 0, (c - g["low"]) / amplitude, 0.5
    )
    # Relatif au marché : bêta 60 j calculé à la main (rolling.cov a des pièges)
    rm = g["sp500_ret_1_m"]
    mean_r = r.rolling(BETA_N, min_periods=BETA_N).mean()
    mean_m = rm.rolling(BETA_N, min_periods=BETA_N).mean()
    mean_rm = (r * rm).rolling(BETA_N, min_periods=BETA_N).mean()
    mean_m2 = (rm * rm).rolling(BETA_N, min_periods=BETA_N).mean()
    cov = mean_rm - mean_r * mean_m
    var = (mean_m2 - mean_m * mean_m).replace(0.0, np.nan)
    out["beta_60"] = cov / var
    return out


def compute_features(prices: pd.DataFrame, market: pd.DataFrame,
                     feature_set: str = DEFAULT_SET) -> pd.DataFrame:
    """Toutes les features, puis rang centile par date (suffixe _rk)."""
    mf = market_features(market)
    merged = prices.merge(
        mf.rename(columns={"sp500_ret_1": "sp500_ret_1_m"})[["date", "sp500_ret_1_m"]],
        on="date", how="left",
    ).sort_values(["ticker", "date"])

    feats = pd.concat(
        [_ticker_features(g, tk) for tk, g in merged.groupby("ticker", sort=False)],
        ignore_index=True,
    )

    # Fenêtres incomplètes (début d'historique de chaque ticker) : suppression documentée.
    avant = len(feats)
    feature_columns(feature_set)  # valide le nom du jeu (lève ValueError sinon)
    sel = FEATURE_SETS[feature_set]
    sel_rk = [f + "_rk" for f in sel]
    feats = feats.dropna(subset=sel).reset_index(drop=True)

    # Normalisation transversale : rang centile parmi les actions du même jour.
    feats[sel_rk] = feats.groupby("date")[sel].rank(pct=True)

    # Contexte de marché (brut, identique pour toutes les actions du jour).
    feats = feats.merge(mf[["date"] + MARKET_FEATURES], on="date", how="left")
    feats = feats.dropna(subset=MARKET_FEATURES).reset_index(drop=True)
    # (attrs posé en dernier : un merge/dropna intermédiaire l'effacerait)
    feats = feats[["date", "ticker"] + sel_rk + MARKET_FEATURES]
    feats.attrs["lignes_fenetres_incompletes"] = avant - len(feats)
    return feats


# ----------------------------------------------------------------------------- label
def build_labels(prices: pd.DataFrame, calendar: pd.Series) -> pd.DataFrame:
    """Label de la section 3.3 du CDC, avec garde-fou calendrier.

    Une ligne (t, ticker) n'est étiquetable que si la date suivante du ticker
    est exactement la séance suivante du marché. Sinon (trou de cotation,
    ex. IBKR oct. 2019) le "lendemain" couvrirait plusieurs séances et le
    label serait faux -> label NaN, compté dans le journal.
    """
    cal = pd.Series(sorted(pd.to_datetime(calendar).unique()))
    next_session = dict(zip(cal[:-1], cal[1:]))

    p = prices.sort_values(["ticker", "date"]).copy()
    g = p.groupby("ticker", sort=False)
    p["next_date"] = g["date"].shift(-1)
    p["ret_next"] = g["close"].shift(-1) / p["close"] - 1
    p["attendu"] = p["date"].map(next_session)
    garde_ok = p["next_date"].notna() & p["next_date"].eq(p["attendu"])
    p.loc[~garde_ok, "ret_next"] = np.nan
    n_garde = int((p["next_date"].notna() & ~garde_ok.where(p["next_date"].notna(), True)).sum())

    # Excès vs médiane du jour, parmi les lignes valides du jour.
    p["excess_ret"] = p["ret_next"] - p.groupby("date")["ret_next"].transform("median")
    q_hi = p.groupby("date")["excess_ret"].transform(lambda s: s.quantile(LABEL_Q_HI))
    q_lo = p.groupby("date")["excess_ret"].transform(lambda s: s.quantile(LABEL_Q_LO))
    p["label"] = np.where(
        p["excess_ret"].isna(), np.nan,
        np.where(p["excess_ret"] >= q_hi, 1.0, np.where(p["excess_ret"] <= q_lo, 0.0, np.nan)),
    )
    out = p[["date", "ticker", "excess_ret", "label"]].copy()
    out.attrs["lignes_garde_calendrier"] = n_garde
    return out


# ----------------------------------------------------------------------------- validation
def validate(ds: pd.DataFrame) -> list[str]:
    """Contrôles bloquants sur le dataset final."""
    problemes: list[str] = []
    if ds.empty:
        return ["dataset vide"]
    if ds.duplicated(["date", "ticker"]).any():
        problemes.append("doublons (date, ticker)")
    ranked = [c for c in ds.columns if c.endswith("_rk")]
    feats = ranked + MARKET_FEATURES
    if not ranked:
        return ["aucune feature rangée dans le dataset"]
    if ds[feats].isna().any().any():
        cols = ds[feats].isna().sum()
        problemes.append(f"features avec NaN : {cols[cols > 0].to_dict()}")
    rk = ds[ranked]
    if ((rk <= 0) | (rk > 1)).any().any():
        problemes.append("rangs hors de l'intervalle (0, 1]")
    lab = ds["label"].dropna()
    if len(lab) < 1000:
        problemes.append(f"trop peu de lignes étiquetées ({len(lab)})")
    else:
        part_1 = float(lab.mean())
        if not 0.40 <= part_1 <= 0.60:
            problemes.append(f"déséquilibre du label suspect : part de 1 = {part_1:.2f}")
    return problemes


# ----------------------------------------------------------------------------- écriture
def _atomic_write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def run(prices_path: Path = IN_PRICES, market_path: Path = IN_MARKET,
        out_path: Path = OUT_DATASET, info_path: Path = OUT_INFO,
        feature_set: str = DEFAULT_SET) -> dict:
    prices, market = load_inputs(prices_path, market_path)
    cols_feat = feature_columns(feature_set)
    feats = compute_features(prices, market, feature_set)
    labels = build_labels(prices, market["date"])
    ds = feats.merge(labels, on=["date", "ticker"], how="left", validate="one_to_one")

    colonnes = ["date", "ticker"] + cols_feat + ["excess_ret", "label"]
    ds = ds[colonnes].sort_values(["date", "ticker"]).reset_index(drop=True)

    problemes = validate(ds)
    lab = ds["label"].dropna()
    info = {
        "genere_le": pd.Timestamp.now().isoformat(timespec="seconds"),
        "periode": [str(ds["date"].min().date()), str(ds["date"].max().date())],
        "lignes": int(len(ds)),
        "lignes_etiquetees": int(len(lab)),
        "part_label_1": round(float(lab.mean()), 4) if len(lab) else None,
        "lignes_zone_grise_ou_sans_lendemain": int(ds["label"].isna().sum()),
        "lignes_fenetres_incompletes_retirees": int(feats.attrs.get("lignes_fenetres_incompletes", 0)),
        "lignes_exclues_garde_calendrier": int(labels.attrs.get("lignes_garde_calendrier", 0)),
        "jeu_de_features": feature_set,
        "nb_features": len(cols_feat),
        "features": cols_feat,
        "entrees_sha256": {"prices": _sha256(Path(prices_path)), "market": _sha256(Path(market_path))},
        "problemes": problemes,
        "ok": not problemes,
    }
    if problemes:
        # on n'écrase pas un dataset sain avec un dataset douteux
        print("REFUSÉ :", " ; ".join(problemes))
    else:
        _atomic_write(ds, Path(out_path))
    Path(info_path).parent.mkdir(parents=True, exist_ok=True)
    Path(info_path).write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return info


def load_dataset(out_path: Path = OUT_DATASET, info_path: Path = OUT_INFO) -> pd.DataFrame:
    """Chargeur pour l'entraînement : refuse un dataset absent ou invalide."""
    if not Path(info_path).exists() or not Path(out_path).exists():
        raise FileNotFoundError("dataset absent — lancer : python -m src.features")
    info = json.loads(Path(info_path).read_text(encoding="utf-8"))
    if not info.get("ok"):
        raise ValueError(f"dataset invalide : {info.get('problemes')}")
    return pd.read_parquet(out_path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Construction des features et du label")
    ap.add_argument("--prices", default=str(IN_PRICES))
    ap.add_argument("--market", default=str(IN_MARKET))
    ap.add_argument("--out", default=str(OUT_DATASET))
    ap.add_argument("--info", default=str(OUT_INFO))
    ap.add_argument("--set", dest="feature_set", choices=sorted(FEATURE_SETS), default=DEFAULT_SET,
                    help="core = 13 features (défaut), full = 24 (CDC complet)")
    a = ap.parse_args(argv)
    try:
        info = run(Path(a.prices), Path(a.market), Path(a.out), Path(a.info), a.feature_set)
    except (ValueError, FileNotFoundError, SystemExit) as e:
        print(f"ERREUR : {e}")
        return 2
    except Exception as e:  # noqa: BLE001
        print(f"PLANTAGE : {type(e).__name__}: {e}")
        return 1
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0 if info["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())