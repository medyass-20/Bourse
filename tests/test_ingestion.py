"""Tests de l'ingestion. Aucun acces reseau : yfinance est remplace par un faux telechargeur."""
import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src import ingestion as ing

NY = ing.NY


def fake_frame(tickers, start, end_date, scale=1.0, missing=()):
    idx = pd.bdate_range(start, end_date, tz="America/New_York")
    cols = pd.MultiIndex.from_product([[t for t in tickers if t not in missing], ["Open", "High", "Low", "Close", "Volume"]])
    rng = np.random.default_rng(0)
    base = rng.uniform(50, 150, len(cols)) * 0 + 100 * scale
    df = pd.DataFrame(np.tile(base, (len(idx), 1)), index=idx, columns=cols)
    df.loc[:, (slice(None), "Volume")] = 1_000_000.0
    return df


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(ing, "RAW_PRICES", tmp_path / "prices" / "prices.parquet")
    monkeypatch.setattr(ing, "RAW_MARKET", tmp_path / "market" / "market.parquet")
    monkeypatch.setattr(ing, "REPORT", tmp_path / "last.json")
    monkeypatch.setattr(ing.time, "sleep", lambda s: None)
    tf = tmp_path / "t.txt"
    tf.write_text("# commentaire\nAAA\nBBB\nCCC\nDDD\nEEE\nFFF\nGGG\nHHH\nIII\nJJJ\n")
    state = {"end": "2026-11-06", "scale": 1.0, "missing": set(), "calls": 0, "fail_first": 0}

    def fake(tickers, start, end=None):
        state["calls"] += 1
        if state["calls"] <= state["fail_first"]:
            raise ConnectionError("reseau coupe")
        return fake_frame(tickers, start, state["end"], state["scale"], state["missing"])

    monkeypatch.setattr(ing, "_yf_download", fake)
    state["tickers_file"] = tf
    state["now"] = datetime(2026, 11, 7, 12, 0, tzinfo=NY)  # samedi : marche ferme
    return state


def test_load_tickers_ignores_comments(env):
    assert ing.load_tickers(env["tickers_file"])[0] == "AAA"
    assert len(ing.load_tickers(env["tickers_file"])) == 10


def test_flatten_multiindex_and_single_ticker():
    raw = fake_frame(["AAA", "BBB"], "2026-01-01", "2026-01-10")
    out = ing.flatten(raw, ["AAA", "BBB", "ZZZ"])
    assert set(out["ticker"]) == {"AAA", "BBB"}
    assert list(out.columns) == ing.PRICE_COLS
    assert out["date"].dt.tz is None
    single = raw["AAA"]                       # colonnes a un seul niveau
    assert set(ing.flatten(single, ["AAA"])["ticker"]) == {"AAA"}


def test_flatten_swapped_levels():
    raw = fake_frame(["AAA"], "2026-01-01", "2026-01-10").swaplevel(axis=1)
    assert set(ing.flatten(raw, ["AAA"])["ticker"]) == {"AAA"}


def test_missing_ticker_reported_and_does_not_block(env):
    env["missing"] = {"CCC"}
    rep = ing.run(tickers_file=env["tickers_file"], now=env["now"])
    assert rep["tickers_manquants"] == ["CCC"] and rep["ok"] is True
    stored = pd.read_parquet(ing.RAW_PRICES)
    assert "CCC" not in set(stored["ticker"]) and stored["ticker"].nunique() == 9


def test_too_many_missing_flags_not_ok(env):
    env["missing"] = {"AAA", "BBB", "CCC"}
    assert ing.run(tickers_file=env["tickers_file"], now=env["now"])["ok"] is False


def test_retries_then_success(env):
    env["fail_first"] = 2
    rep = ing.run(tickers_file=env["tickers_file"], now=env["now"])
    assert rep["ok"] and rep["tickers_stockes"] == 10


def test_output_format_matches_contract(env):
    ing.run(tickers_file=env["tickers_file"], now=env["now"])
    p, m = pd.read_parquet(ing.RAW_PRICES), pd.read_parquet(ing.RAW_MARKET)
    assert list(p.columns) == ["date", "ticker", "open", "high", "low", "close", "volume"]
    assert list(m.columns) == ["date", "sp500", "vix"]
    assert not p.duplicated(["date", "ticker"]).any() and not m.duplicated("date").any()
    assert p["date"].is_monotonic_increasing or p.sort_values(["ticker", "date"]).equals(p)


def test_incremental_keeps_history_and_adds_new_day(env):
    env["end"] = "2026-11-04"
    ing.run(tickers_file=env["tickers_file"], now=env["now"])
    n1 = len(pd.read_parquet(ing.RAW_PRICES))
    first_date = pd.read_parquet(ing.RAW_PRICES)["date"].min()
    env["end"] = "2026-11-06"
    ing.run(tickers_file=env["tickers_file"], now=env["now"])
    after = pd.read_parquet(ing.RAW_PRICES)
    assert len(after) == n1 + 10 * 2            # 2 nouvelles seances x 10 tickers
    assert after["date"].min() == first_date     # historique intact
    assert not after.duplicated(["date", "ticker"]).any()


def test_adjustment_detected_on_overlap(env):
    ing.run(tickers_file=env["tickers_file"], now=env["now"])
    env["scale"] = 0.5                            # simule un split : tout le passe change
    rep = ing.run(tickers_file=env["tickers_file"], now=env["now"])
    assert rep.get("tickers_ajustes") and any("full-refresh" in w for w in rep["avertissements"])


def test_full_refresh_replaces_history_when_ok(env):
    ing.run(tickers_file=env["tickers_file"], now=env["now"])
    env["scale"] = 0.5
    ing.run(full_refresh=True, tickers_file=env["tickers_file"], now=env["now"])
    assert (pd.read_parquet(ing.RAW_PRICES)["close"] == 50).all()


def test_full_refresh_does_not_overwrite_when_download_broken(env):
    ing.run(tickers_file=env["tickers_file"], now=env["now"])
    before = pd.read_parquet(ing.RAW_PRICES)
    env["missing"] = set(ing.load_tickers(env["tickers_file"]))   # Yahoo ne renvoie plus rien
    rep = ing.run(full_refresh=True, tickers_file=env["tickers_file"], now=env["now"])
    assert rep["ok"] is False
    pd.testing.assert_frame_equal(before, pd.read_parquet(ing.RAW_PRICES))


def test_today_row_dropped_before_17h_ny(env):
    env["end"] = "2026-11-06"
    df = ing.flatten(fake_frame(["AAA"], "2026-11-01", "2026-11-06"), ["AAA"])
    avant = ing.drop_unfinished_today(df, datetime(2026, 11, 6, 12, 0, tzinfo=NY))
    apres = ing.drop_unfinished_today(df, datetime(2026, 11, 6, 17, 30, tzinfo=NY))
    assert avant["date"].max() == pd.Timestamp("2026-11-05")
    assert apres["date"].max() == pd.Timestamp("2026-11-06")


def test_stale_data_warning(env):
    env["end"] = "2026-10-20"
    rep = ing.run(tickers_file=env["tickers_file"], now=env["now"])
    assert any("trop ancienne" in w for w in rep["avertissements"])


def test_report_written(env):
    ing.run(tickers_file=env["tickers_file"], now=env["now"])
    rep = json.loads(ing.REPORT.read_text())
    assert rep["tickers_stockes"] == 10 and rep["derniere_date"] == "2026-11-06"