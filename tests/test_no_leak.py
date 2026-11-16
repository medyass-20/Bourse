# -*- coding: utf-8 -*-
"""Tests anti-fuite et de formules pour src/features.py.

Principe central : une feature du jour t ne doit dépendre que des données <= t.
Le test le plus fort est la TRONCATURE : on recalcule tout en coupant
l'historique après T ; les valeurs <= T doivent être identiques au calcul
sur l'historique complet. Toute fuite échoue à ce test, y compris via ewm.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (
    FEATURE_SETS,
    MARKET_FEATURES,
    build_labels,
    compute_features,
    feature_columns,
    load_inputs,
    run,
    validate,
)

SETS = ["core", "full"]
ALL_FEATURES = feature_columns("full")          # 24 colonnes
RANKED = [c for c in ALL_FEATURES if c.endswith("_rk")]   # 20 colonnes rangées

RNG = np.random.default_rng(7)
N_TICKERS = 12
N_DAYS = 250  # > 60 (bêta) + marge pour dépasser le seuil de lignes étiquetées


def _donnees(n_days: int = N_DAYS, n_tickers: int = N_TICKERS):
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    rows = []
    for i, tk in enumerate(tickers):
        close = 100 * (1 + i / 10) * np.exp(np.cumsum(RNG.normal(0.0003, 0.015, n_days)))
        openp = close * (1 + RNG.normal(0, 0.004, n_days))
        high = np.maximum(openp, close) * (1 + np.abs(RNG.normal(0, 0.004, n_days)))
        low = np.minimum(openp, close) * (1 - np.abs(RNG.normal(0, 0.004, n_days)))
        vol = RNG.integers(1_000_000, 5_000_000, n_days).astype(float)
        rows.append(pd.DataFrame({
            "date": dates, "ticker": tk, "open": openp, "high": high,
            "low": low, "close": close, "volume": vol,
        }))
    prices = pd.concat(rows, ignore_index=True)
    sp500 = 5000 * np.exp(np.cumsum(RNG.normal(0.0002, 0.01, n_days)))
    vix = np.clip(18 + np.cumsum(RNG.normal(0, 0.5, n_days)), 10, 60)
    market = pd.DataFrame({"date": dates, "sp500": sp500, "vix": vix})
    return prices, market


@pytest.fixture(scope="module")
def donnees():
    return _donnees()


# --------------------------------------------------------------- 1. troncature
@pytest.mark.parametrize("jeu", SETS)
def test_troncature_features_inchangees(donnees, jeu):
    """Couper l'historique après T ne change aucune feature <= T (pas de fuite)."""
    prices, market = donnees
    T = prices["date"].sort_values().unique()[-20]
    cols = feature_columns(jeu)
    complet = compute_features(prices, market, jeu)
    tronque = compute_features(
        prices[prices["date"] <= T], market[market["date"] <= T], jeu
    )
    a = complet[complet["date"] <= T].set_index(["date", "ticker"])[cols].sort_index()
    b = tronque.set_index(["date", "ticker"])[cols].sort_index()
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-10)


@pytest.mark.parametrize("jeu", SETS)
def test_modification_futur_sans_effet(donnees, jeu):
    """Multiplier les prix après T par 3 ne change rien <= T."""
    prices, market = donnees
    T = prices["date"].sort_values().unique()[-10]
    cols = feature_columns(jeu)
    complet = compute_features(prices, market, jeu)
    p2, m2 = prices.copy(), market.copy()
    p2.loc[p2["date"] > T, ["open", "high", "low", "close"]] *= 3.0
    m2.loc[m2["date"] > T, "sp500"] *= 3.0
    modifie = compute_features(p2, m2, jeu)
    a = complet[complet["date"] <= T].set_index(["date", "ticker"])[cols].sort_index()
    b = modifie[modifie["date"] <= T].set_index(["date", "ticker"])[cols].sort_index()
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-10)


# --------------------------------------------------------------- 2. le label voit t+1 et rien d'autre
def test_label_depend_du_lendemain(donnees):
    prices, market = donnees
    base = build_labels(prices, market["date"])
    dates = sorted(prices["date"].unique())
    t, t1 = dates[-30], dates[-29]
    p2 = prices.copy()
    # On fait exploser T00 le lendemain : son excès au jour t doit devenir top 30 %.
    p2.loc[(p2["ticker"] == "T00") & (p2["date"] == t1), "close"] *= 1.5
    apres = build_labels(p2, market["date"])
    lab = apres[(apres["date"] == t) & (apres["ticker"] == "T00")]["label"].iloc[0]
    assert lab == 1.0
    # Les jours strictement avant t sont intouchés.
    a = base[base["date"] < t].set_index(["date", "ticker"])[["excess_ret", "label"]]
    b = apres[apres["date"] < t].set_index(["date", "ticker"])[["excess_ret", "label"]]
    pd.testing.assert_frame_equal(a.sort_index(), b.sort_index())


def test_garde_calendrier_trou_de_cotation(donnees):
    """Si un ticker saute une séance, la ligne d'avant le trou n'est pas étiquetable."""
    prices, market = donnees
    dates = sorted(prices["date"].unique())
    t_avant, t_trou = dates[80], dates[81]
    p2 = prices[~((prices["ticker"] == "T03") & (prices["date"] == t_trou))]
    labels = build_labels(p2, market["date"])
    ligne = labels[(labels["ticker"] == "T03") & (labels["date"] == t_avant)]
    assert ligne["label"].isna().all() and ligne["excess_ret"].isna().all()
    assert labels.attrs["lignes_garde_calendrier"] >= 1


def test_derniere_date_sans_label(donnees):
    prices, market = donnees
    labels = build_labels(prices, market["date"])
    derniere = labels[labels["date"] == labels["date"].max()]
    assert derniere["label"].isna().all()


# --------------------------------------------------------------- 3. formules vérifiées à la main
def test_formule_ret_5(donnees):
    prices, market = donnees
    feats = compute_features(prices, market, "full")
    tk = prices[prices["ticker"] == "T01"].sort_values("date").reset_index(drop=True)
    i = 100
    attendu = tk.loc[i, "close"] / tk.loc[i - 5, "close"] - 1
    # ret_5 brut n'est plus dans la sortie (rangé) : on le recalcule via le rang.
    jour = feats[feats["date"] == tk.loc[i, "date"]]
    brut = {}
    for t in jour["ticker"]:
        s = prices[(prices["ticker"] == t)].sort_values("date").set_index("date")["close"]
        brut[t] = s.loc[tk.loc[i, "date"]] / s.shift(5).loc[tk.loc[i, "date"]] - 1
    rang_attendu = pd.Series(brut).rank(pct=True)["T01"]
    obtenu = jour[jour["ticker"] == "T01"]["ret_5_rk"].iloc[0]
    assert obtenu == pytest.approx(rang_attendu)
    assert brut["T01"] == pytest.approx(attendu)


def test_close_pos_bornes_et_seance_plate():
    prices, market = _donnees(90, 4)
    # une séance plate : high == low
    i = prices.index[50]
    prices.loc[i, ["high", "low", "open", "close"]] = 100.0
    from src.features import _ticker_features  # noqa: PLC0415
    tk = prices[prices["ticker"] == prices.loc[i, "ticker"]].sort_values("date").copy()
    tk["sp500_ret_1_m"] = 0.0
    out = _ticker_features(tk, tk["ticker"].iloc[0])
    assert out["close_pos"].dropna().between(0, 1).all()
    assert out.loc[out["date"] == prices.loc[i, "date"], "close_pos"].iloc[0] == 0.5


def test_rsi_bornes(donnees):
    prices, _ = donnees
    from src.features import _rsi  # noqa: PLC0415
    s = prices[prices["ticker"] == "T00"].sort_values("date")["close"]
    rsi = _rsi(s).dropna()
    assert rsi.between(0, 100).all()
    # hausse pure -> RSI = 100
    up = pd.Series(np.arange(1.0, 40.0))
    assert _rsi(up).dropna().iloc[-1] == pytest.approx(100.0)


# --------------------------------------------------------------- 4. rangs et équilibre
def test_rangs_dans_0_1_et_par_jour(donnees):
    prices, market = donnees
    feats = compute_features(prices, market, "full")
    rk = feats[RANKED]
    assert ((rk > 0) & (rk <= 1)).all().all()
    # le rang max de chaque jour vaut 1 (le meilleur du jour)
    assert (feats.groupby("date")["ret_1_rk"].max() == 1.0).all()


def test_equilibre_label(donnees):
    prices, market = donnees
    labels = build_labels(prices, market["date"])
    lab = labels["label"].dropna()
    assert 0.40 <= lab.mean() <= 0.60
    # ~60 % des lignes valides sont étiquetées (30+30), zone grise exclue
    valides = labels["excess_ret"].notna().sum()
    assert 0.5 <= len(lab) / valides <= 0.75


# --------------------------------------------------------------- 5. contrat d'entrée et pipeline complet
def test_contrat_refuse_doublons(tmp_path, donnees):
    prices, market = donnees
    mauvais = pd.concat([prices, prices.iloc[[0]]], ignore_index=True)
    pp, mm = tmp_path / "p.parquet", tmp_path / "m.parquet"
    mauvais.to_parquet(pp, index=False)
    market.to_parquet(mm, index=False)
    with pytest.raises(ValueError, match="doublons"):
        load_inputs(pp, mm)


def test_contrat_refuse_dates_hors_calendrier(tmp_path, donnees):
    prices, market = donnees
    pp, mm = tmp_path / "p.parquet", tmp_path / "m.parquet"
    prices.to_parquet(pp, index=False)
    market.iloc[:-5].to_parquet(mm, index=False)  # market ampute de 5 séances
    with pytest.raises(ValueError, match="absentes de market"):
        load_inputs(pp, mm)


@pytest.mark.parametrize("jeu", SETS)
def test_run_complet(tmp_path, donnees, jeu):
    prices, market = donnees
    pp, mm = tmp_path / "p.parquet", tmp_path / "m.parquet"
    out, info = tmp_path / "dataset.parquet", tmp_path / "info.json"
    prices.to_parquet(pp, index=False)
    market.to_parquet(mm, index=False)
    res = run(pp, mm, out, info, jeu)
    assert res["ok"], res["problemes"]
    ds = pd.read_parquet(out)
    attendu = feature_columns(jeu)
    assert list(ds.columns) == ["date", "ticker"] + attendu + ["excess_ret", "label"]
    assert res["nb_features"] == len(attendu) == {"core": 13, "full": 24}[jeu]
    assert res["jeu_de_features"] == jeu
    assert validate(ds) == []
    assert res["lignes_etiquetees"] > 0
    assert res["entrees_sha256"]["prices"]


def test_dataset_refuse_si_invalide(tmp_path, donnees):
    """Si la validation échoue, on n'écrase pas le dataset existant."""
    prices, market = donnees
    pp, mm = tmp_path / "p.parquet", tmp_path / "m.parquet"
    out, infop = tmp_path / "dataset.parquet", tmp_path / "info.json"
    prices.to_parquet(pp, index=False)
    market.to_parquet(mm, index=False)
    run(pp, mm, out, infop)
    bon = out.read_bytes()
    # entrée trop courte : aucune ligne ne survit aux fenêtres -> refus
    courtes_p = prices[prices["date"] <= prices["date"].unique()[30]]
    courtes_m = market[market["date"] <= prices["date"].unique()[30]]
    courtes_p.to_parquet(pp, index=False)
    courtes_m.to_parquet(mm, index=False)
    res = run(pp, mm, out, infop)
    assert not res["ok"]
    assert out.read_bytes() == bon  # l'ancien dataset sain est intact
    from src.features import load_dataset  # noqa: PLC0415
    with pytest.raises(ValueError, match="invalide"):
        load_dataset(out, infop)


# --------------------------------------------------------------- 6. jeux de features
def test_core_a_13_features_et_full_a_24():
    assert len(feature_columns("core")) == 13
    assert len(feature_columns("full")) == 24
    # core est un sous-ensemble strict de full
    assert set(feature_columns("core")) < set(feature_columns("full"))
    assert set(MARKET_FEATURES) <= set(feature_columns("core"))


def test_jeu_inconnu_refuse():
    with pytest.raises(ValueError, match="jeu inconnu"):
        feature_columns("nimportequoi")


def test_core_et_full_donnent_memes_valeurs_communes(donnees):
    """Une feature commune aux deux jeux a la même valeur : le jeu choisi
    ne change pas le calcul (les rangs d'un jour ne dépendent que de ce jour)."""
    prices, market = donnees
    core = compute_features(prices, market, "core").set_index(["date", "ticker"])
    full = compute_features(prices, market, "full").set_index(["date", "ticker"])
    commun = [c for c in feature_columns("core") if c in full.columns]
    idx = core.index.intersection(full.index)
    pd.testing.assert_frame_equal(
        core.loc[idx, commun].sort_index(), full.loc[idx, commun].sort_index(),
        check_exact=False, rtol=1e-10,
    )
    # core garde plus de lignes (sa fenêtre la plus longue est 60 j, celle de full aussi
    # mais full exige aussi MM50/MACD) : jamais moins.
    assert len(core) >= len(full)