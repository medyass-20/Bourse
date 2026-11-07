"""Ingestion (Role 1 - Donnees) : recupere les cours via yfinance et les STOCKE BRUTS.

Ce module ne nettoie rien et ne calcule aucune variable (c'est le travail des autres modules).
Il garantit seulement : format de sortie fige, pas de doublons, pas d'historique perdu, erreurs signalees.

Sorties (contrat d'equipe) :
  data/raw/prices/prices.parquet : date, ticker, open, high, low, close, volume   (auto_adjust=True)
  data/raw/market/market.parquet : date, sp500, vix
  data/raw/last_ingestion.json   : compte rendu du dernier passage (tickers manquants, alertes...)

Utilisation :
  python -m src.ingestion                  # premiere fois : historique complet ; ensuite : mise a jour incrementale
  python -m src.ingestion --full-refresh   # a faire environ 1 fois par mois (les prix ajustes se reecrivent)
  python -m src.ingestion --fallback       # tente Stooq pour les tickers absents de Yahoo
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PRICES = ROOT / "data" / "raw" / "prices" / "prices.parquet"
RAW_MARKET = ROOT / "data" / "raw" / "market" / "market.parquet"
REPORT = ROOT / "data" / "raw" / "last_ingestion.json"
TICKERS_FILE = ROOT / "tickers.txt"

PRICE_COLS = ["date", "ticker", "open", "high", "low", "close", "volume"]
MARKET_COLS = ["date", "sp500", "vix"]
OHLCV = ["Open", "High", "Low", "Close", "Volume"]
MARKET_TICKERS = {"^GSPC": "sp500", "^VIX": "vix"}

HISTORY_YEARS = 11          # 10 ans minimum + marge pour la chauffe des variables
BATCH_SIZE = 25             # tickers par lot
BATCH_PAUSE = 2.0           # secondes entre deux lots
RETRIES = 3
OVERLAP_DAYS = 10           # recouvrement des mises a jour incrementales
MAX_MISSING_SHARE = 0.10    # au-dela, un --full-refresh n'ecrase pas l'ancien fichier
STALE_DAYS = 4              # derniere date trop ancienne -> alerte
ADJ_TOLERANCE = 1e-3        # ecart relatif de cloture signalant un ajustement (split/dividende)
NY = ZoneInfo("America/New_York")
CLOSE_HOUR_NY = 17          # le script ne considere la seance du jour complete qu'apres 17h a New York

log = logging.getLogger("ingestion")


# --------------------------------------------------------------------------- utilitaires
def load_tickers(path: Path = TICKERS_FILE) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    out = [x.strip().upper() for x in lines if x.strip() and not x.lstrip().startswith("#")]
    if len(out) != len(set(out)):
        raise ValueError("doublons dans la liste de tickers")
    return out


def _yf_download(tickers, start, end=None):
    """Appel unique a yfinance (isole pour pouvoir le remplacer dans les tests)."""
    import yfinance as yf

    return yf.download(
        tickers=tickers, start=start, end=end, interval="1d", auto_adjust=True,
        group_by="ticker", multi_level_index=True, progress=False, threads=True,
    )


def _with_retries(fn, tries: int = RETRIES, wait: float = 3.0):
    last = None
    for k in range(1, tries + 1):
        try:
            res = fn()
            if res is None or len(res) == 0:
                raise RuntimeError("reponse vide")
            return res
        except Exception as e:  # noqa: BLE001 - reseau/Yahoo : on reessaie quel que soit le type
            last = e
            log.warning("tentative %d/%d echouee : %s", k, tries, e)
            if k < tries:
                time.sleep(wait * k)
    raise RuntimeError(f"echec apres {tries} tentatives : {last}")


def _one_ticker(part: pd.DataFrame, ticker: str) -> pd.DataFrame:
    part = part.reindex(columns=OHLCV)
    part = part.dropna(how="all")            # lignes entierement vides = pas de cotation, pas une donnee
    if part.empty:
        return pd.DataFrame(columns=PRICE_COLS)
    out = part.copy()
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    out.index.name = "date"
    out = out.reset_index()
    out.columns = [str(c).lower() for c in out.columns]
    out["ticker"] = ticker
    return out[PRICE_COLS]


def flatten(raw: pd.DataFrame | None, tickers: list[str]) -> pd.DataFrame:
    """Colonnes yfinance a deux niveaux (Ticker, Prix) -> format long. Gere aussi un lot a un seul ticker."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=PRICE_COLS)
    frames = []
    if isinstance(raw.columns, pd.MultiIndex):
        l0, l1 = set(raw.columns.get_level_values(0)), set(raw.columns.get_level_values(1))
        if not (set(tickers) & l0) and (set(tickers) & l1):   # niveaux inverses (Prix, Ticker)
            raw = raw.swaplevel(axis=1)
            l0 = set(raw.columns.get_level_values(0))
        for t in tickers:
            if t in l0:
                frames.append(_one_ticker(raw[t], t))
    elif len(tickers) == 1:
        frames.append(_one_ticker(raw, tickers[0]))
    else:
        log.error("format inattendu : colonnes a un niveau pour %d tickers", len(tickers))
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PRICE_COLS)


# --------------------------------------------------------------------------- telechargement
def fetch_prices(tickers: list[str], start: str, end: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Telecharge par lots. Un ticker introuvable est signale et ignore, sans bloquer le reste."""
    frames: list[pd.DataFrame] = []
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i : i + BATCH_SIZE]
        log.info("lot %d/%d : %d tickers", i // BATCH_SIZE + 1, -(-len(tickers) // BATCH_SIZE), len(batch))
        try:
            raw = _with_retries(lambda b=batch: _yf_download(b, start, end))
            frames.append(flatten(raw, batch))
        except RuntimeError as e:
            log.error("lot abandonne (%s) : %s", ", ".join(batch[:3]) + "...", e)
        if i + BATCH_SIZE < len(tickers):
            time.sleep(BATCH_PAUSE)
    prices = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PRICE_COLS)
    got = set(prices["ticker"].unique())
    missing = [t for t in tickers if t not in got]
    for t in missing:
        log.warning("ticker absent ou sans seance valide : %s", t)
    return prices, missing


def fetch_market(start: str, end: str | None = None) -> pd.DataFrame:
    raw = _with_retries(lambda: _yf_download(list(MARKET_TICKERS), start, end))
    long = flatten(raw, list(MARKET_TICKERS))
    if long.empty:
        raise RuntimeError("aucune donnee de marche (^GSPC, ^VIX)")
    wide = long.pivot(index="date", columns="ticker", values="close").rename(columns=MARKET_TICKERS)
    for c in ("sp500", "vix"):
        if c not in wide.columns:
            raise RuntimeError(f"serie de marche absente : {c}")
    return wide[["sp500", "vix"]].sort_index().reset_index()[MARKET_COLS]


def fetch_stooq(tickers: list[str], start: str) -> pd.DataFrame:
    """Source de secours (opt-in). Prix non ajustes des dividendes : les melanger a Yahoo introduit un
    ecart, d'ou l'option --fallback desactivee par defaut et le message d'avertissement."""
    import io

    import requests

    out = []
    for t in tickers:
        url = f"https://stooq.com/q/d/l/?s={t.lower()}.us&i=d"
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            d = pd.read_csv(io.StringIO(resp.text))
        except Exception as e:  # noqa: BLE001
            log.warning("stooq %s : %s", t, e)
            continue
        if d.empty or "Close" not in d.columns:
            continue
        d.columns = [c.lower() for c in d.columns]
        d["date"] = pd.to_datetime(d["date"])
        d = d[d["date"] >= pd.Timestamp(start)]
        d["ticker"] = t
        out.append(d[PRICE_COLS])
        time.sleep(0.5)  # politesse envers la source
    if out:
        log.warning("%d ticker(s) pris sur STOOQ (prix non ajustes des dividendes) : %s",
                    len(out), ", ".join(x["ticker"].iloc[0] for x in out))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=PRICE_COLS)


# --------------------------------------------------------------------------- regles de traitement
def drop_unfinished_today(df: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """Avant 17h a New York, la ligne du jour est incomplete : on l'ignore avec un avertissement."""
    now = (now or datetime.now(NY)).astimezone(NY)
    if df.empty or now.hour >= CLOSE_HOUR_NY:
        return df
    today = pd.Timestamp(now.date())
    mask = df["date"] == today
    if mask.any():
        log.warning("il est %s a New York (avant 17h) : ligne du jour ignoree (%d lignes)", now.strftime("%H:%M"), mask.sum())
    return df[~mask]


def detect_adjustments(old: pd.DataFrame, new: pd.DataFrame) -> list[str]:
    """Compare les clotures deja stockees a celles retelechargees sur la zone de recouvrement.
    Un ecart signale un split/dividende : l'historique ancien n'est plus aligne -> --full-refresh conseille."""
    if old.empty or new.empty:
        return []
    m = old.merge(new, on=["date", "ticker"], suffixes=("_old", "_new"))
    if m.empty:
        return []
    rel = (m["close_new"] / m["close_old"] - 1).abs()
    return sorted(m.loc[rel > ADJ_TOLERANCE, "ticker"].unique().tolist())


def merge_keep_history(old: pd.DataFrame, new: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Ajoute le nouveau a l'ancien. En cas de doublon, la ligne la plus recente (nouvelle) gagne."""
    df = pd.concat([old, new], ignore_index=True) if len(old) else new.copy()
    return df.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Ecriture atomique : un plantage en cours d'ecriture ne corrompt jamais le fichier existant."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- orchestration
def run(full_refresh: bool = False, start: str | None = None, tickers_file: Path = TICKERS_FILE,
        fallback: bool = False, now: datetime | None = None) -> dict:
    now = now or datetime.now(NY)
    tickers = load_tickers(tickers_file)
    rep: dict = {"horodatage_ny": now.isoformat(timespec="seconds"), "mode": "full-refresh" if full_refresh else "incremental",
                 "tickers_demandes": len(tickers), "avertissements": [], "ok": True}

    old_p = pd.read_parquet(RAW_PRICES) if RAW_PRICES.exists() else pd.DataFrame(columns=PRICE_COLS)
    old_m = pd.read_parquet(RAW_MARKET) if RAW_MARKET.exists() else pd.DataFrame(columns=MARKET_COLS)
    first = old_p.empty
    default_start = (date.today() - timedelta(days=365 * HISTORY_YEARS)).isoformat()
    if full_refresh or first:
        dl_start = start or default_start
        rep["mode"] = "historique complet" if first and not full_refresh else rep["mode"]
    else:
        dl_start = (pd.to_datetime(old_p["date"]).max() - pd.Timedelta(days=OVERLAP_DAYS)).strftime("%Y-%m-%d")
    rep["debut_telechargement"] = dl_start
    log.info("mode=%s, depuis %s, %d tickers", rep["mode"], dl_start, len(tickers))

    prices, missing = fetch_prices(tickers, dl_start)
    if missing and fallback:
        extra = fetch_stooq(missing, dl_start)
        if not extra.empty:
            prices = pd.concat([prices, extra], ignore_index=True)
            rep["tickers_stooq"] = sorted(extra["ticker"].unique())
            missing = [t for t in missing if t not in set(extra["ticker"])]
    prices = drop_unfinished_today(prices, now)
    rep["tickers_manquants"] = missing

    try:
        market = drop_unfinished_today(fetch_market(dl_start), now)
    except RuntimeError as e:
        log.error("marche non recupere : %s", e)
        market = pd.DataFrame(columns=MARKET_COLS)
        rep["avertissements"].append(f"marche non recupere : {e}")
        rep["ok"] = False

    # --- prix
    too_many_missing = len(missing) > MAX_MISSING_SHARE * len(tickers)
    if too_many_missing:
        rep["avertissements"].append(f"{len(missing)} tickers manquants (> {MAX_MISSING_SHARE:.0%})")
        rep["ok"] = False
    if full_refresh and not first and (too_many_missing or prices.empty):
        log.error("full-refresh incomplet : ancien fichier conserve")
        rep["avertissements"].append("full-refresh abandonne, ancien fichier conserve")
        final_p, final_m = old_p, old_m
    else:
        if not full_refresh and not first:
            adj = detect_adjustments(old_p, prices)
            if adj:
                msg = f"ajustement (split/dividende) detecte pour {len(adj)} ticker(s) : lancer --full-refresh ({', '.join(adj[:8])}...)"
                log.warning(msg)
                rep["avertissements"].append(msg)
                rep["tickers_ajustes"] = adj
        base_p = old_p.iloc[:0] if (full_refresh) else old_p
        final_p = merge_keep_history(base_p, prices, ["ticker", "date"])
        base_m = old_m.iloc[:0] if (full_refresh and not market.empty) else old_m
        final_m = merge_keep_history(base_m, market, ["date"])
        if len(final_p):
            save_parquet(final_p, RAW_PRICES)
        if len(final_m):
            save_parquet(final_m, RAW_MARKET)

    # --- bilan
    if final_p.empty:
        rep["ok"] = False
        rep["avertissements"].append("aucune donnee de prix stockee")
    else:
        last = pd.to_datetime(final_p["date"]).max()
        rep.update(lignes_prix=int(len(final_p)), tickers_stockes=int(final_p["ticker"].nunique()),
                   premiere_date=str(pd.to_datetime(final_p["date"]).min().date()), derniere_date=str(last.date()))
        age = (pd.Timestamp(now.date()) - last).days
        if age > STALE_DAYS:
            msg = f"derniere date trop ancienne : {last.date()} ({age} jours)"
            log.warning(msg)
            rep["avertissements"].append(msg)
    rep["lignes_marche"] = int(len(final_m))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("termine : ok=%s, %d avertissement(s)", rep["ok"], len(rep["avertissements"]))
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Ingestion des cours bruts (yfinance) vers data/raw/")
    ap.add_argument("--full-refresh", action="store_true", help="retelecharger tout l'historique (1 fois par mois)")
    ap.add_argument("--start", help="date de debut AAAA-MM-JJ (par defaut : 11 ans en arriere)")
    ap.add_argument("--tickers-file", default=str(TICKERS_FILE))
    ap.add_argument("--fallback", action="store_true", help="essayer Stooq pour les tickers absents de Yahoo")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        rep = run(a.full_refresh, a.start, Path(a.tickers_file), a.fallback)
    except Exception as e:  # noqa: BLE001
        log.exception("echec de l'ingestion : %s", e)
        return 1
    return 0 if rep["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())