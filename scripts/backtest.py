"""Swing-setup backtest for Nifty 100 / Midcap 150 / Smallcap 250 stocks.

Rules (same as the planned scanner):
  setups   : breakout (20-day high on 1.5x volume from a tight base), pullback (bounce off the
             50-EMA in an uptrend), oversold bounce (RSI turns up from below 30, above 200-SMA)
  filters  : liquidity, weekly uptrend, relative strength top 30%, sector index uptrend,
             avoid list (gap-up > 4%, > 8% above 20-EMA, RSI > 75), market regime
  entry    : next day, only if price trades above the signal day's high (confirmation)
  exits    : target +7.5%, stop at structure (2-4% below entry; wider = skipped),
             stop moves to cost after +4%, time exit after 15 trading days
  costs    : 0.3% per round trip
  portfolio: Rs 2.5L, 1% risk per trade, max 4 open, max 2 per sector, pause after -6% month
Outputs backtest.json and a Markdown summary.
"""
import io
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IST = timezone(timedelta(hours=5, minutes=30))
P = dict(target=0.075, max_risk=0.04, min_risk=0.02, be_at=0.04, max_days=15, cost=0.003,
         capital=250000, risk=0.01, max_open=4, max_sector=2, month_stop=-0.06, pos_cap=0.25,
         min_turnover=5e7, min_price=20, rs_min=70)
LISTS = {"Large": "ind_nifty100list.csv", "Mid": "ind_niftymidcap150list.csv", "Small": "ind_niftysmallcap250list.csv"}
SECTOR_INDEX = {
    "Information Technology": "^CNXIT", "Financial Services": "NIFTY_FIN_SERVICE.NS",
    "Automobile and Auto Components": "^CNXAUTO", "Fast Moving Consumer Goods": "^CNXFMCG",
    "Healthcare": "^CNXPHARMA", "Metals & Mining": "^CNXMETAL", "Oil Gas & Consumable Fuels": "^CNXENERGY",
    "Power": "^CNXENERGY", "Realty": "^CNXREALTY", "Capital Goods": "^CNXINFRA", "Construction": "^CNXINFRA",
    "Construction Materials": "^CNXINFRA", "Telecommunication": "^CNXINFRA", "Media Entertainment & Publication": "^CNXMEDIA",
}
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
           "Accept": "text/csv,*/*", "Accept-Language": "en-US,en;q=0.9"}


# ------------------------------------------------------------------ data
def load_universe():
    import requests
    frames = []
    for cap, f in LISTS.items():
        df = None
        for url in (f"https://www.niftyindices.com/IndexConstituent/{f}",
                    f"https://nsearchives.nseindia.com/content/indices/{f}",
                    f"https://archives.nseindia.com/content/indices/{f}"):
            try:
                r = requests.get(url, headers=HEADERS, timeout=30)
                if r.status_code == 200 and b"Symbol" in r.content[:500]:
                    df = pd.read_csv(io.BytesIO(r.content))
                    break
                print(f"{url}: HTTP {r.status_code}")
            except Exception as e:
                print(f"{url}: {e}")
        if df is None:
            cache = ROOT / "scripts" / "universe_cache.csv"
            if cache.exists():
                print(f"Using cached universe for {cap}")
                c = pd.read_csv(cache)
                frames.append(c[c["cap"] == cap])
                continue
            sys.exit(f"Could not download the {cap} cap list ({f}).")
        df.columns = [c.strip() for c in df.columns]
        frames.append(pd.DataFrame({"symbol": df["Symbol"].str.strip(), "industry": df["Industry"].str.strip(), "cap": cap}))
    uni = pd.concat(frames).drop_duplicates("symbol")
    uni.to_csv(ROOT / "scripts" / "universe_cache.csv", index=False)
    print(f"Universe: {uni['cap'].value_counts().to_dict()}")
    return uni


def download(tickers, period="6y"):
    import yfinance as yf
    out = {}
    for i in range(0, len(tickers), 100):
        chunk = tickers[i:i + 100]
        raw = yf.download(chunk, period=period, interval="1d", group_by="ticker", auto_adjust=True,
                          progress=False, threads=True)
        for t in chunk:
            if t in raw.columns.get_level_values(0):
                d = raw[t].dropna(subset=["Close"])
                if len(d) > 260:
                    out[t] = d
        print(f"downloaded {min(i + 100, len(tickers))}/{len(tickers)}")
    return out


def wide(data, field, index):
    return pd.DataFrame({t: d[field] for t, d in data.items()}).reindex(index)


def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


# ------------------------------------------------------------------ signals
def build_signals(px, nifty, sector_close, uni):
    """All frames are dates x symbols. Everything uses data up to and including the signal day."""
    C, O, H, L, V = px["Close"], px["Open"], px["High"], px["Low"], px["Volume"]
    ema20, ema50 = C.ewm(span=20, adjust=False).mean(), C.ewm(span=50, adjust=False).mean()
    sma200 = C.rolling(200).mean()
    R = rsi(C)
    avgvol = V.rolling(20).mean().shift(1)
    turnover = (C * V).rolling(20).mean()
    hh20 = H.rolling(20).max().shift(1)
    base = (H.rolling(15).max() - L.rolling(15).min()).shift(1) / C.shift(1)
    high252 = H.rolling(252, min_periods=120).max()

    # weekly trend from the last COMPLETED week
    W = C.resample("W-FRI").last()
    wup = ((W > W.rolling(30).mean()) & (W.rolling(10).mean() > W.rolling(30).mean())).shift(1)
    wup = wup.reindex(C.index, method="ffill").fillna(False).astype(bool)

    n = nifty.reindex(C.index).ffill()
    rel = lambda k: (C / C.shift(k) - 1).sub(n / n.shift(k) - 1, axis=0)
    rs = ((rel(63).rank(axis=1, pct=True) + rel(126).rank(axis=1, pct=True)) / 2 * 100)

    sec_up = pd.DataFrame(True, index=C.index, columns=C.columns)
    for sym in C.columns:
        ind = uni.loc[sym, "industry"] if sym in uni.index else None
        sc = sector_close.get(SECTOR_INDEX.get(ind))
        if sc is not None:
            s = sc.reindex(C.index).ffill()
            sec_up[sym] = (s > s.ewm(span=50, adjust=False).mean()).fillna(True)

    breadth = (C > ema50).sum(axis=1) / C.notna().sum(axis=1)
    regime = (n > n.ewm(span=50, adjust=False).mean()) & (breadth >= 0.5)

    liquid = (turnover >= P["min_turnover"]) & (C >= P["min_price"])
    gap = O / C.shift(1) - 1 > 0.04
    avoid = gap | (C / ema20 - 1 > 0.08) | (R > 75)
    quality = wup & (rs >= P["rs_min"]) & sec_up
    vr = V / avgvol

    setups = {
        "Breakout": ((C > hh20) & (vr >= 1.5) & (base <= 0.15) & (C > ema50), L.rolling(5).min()),
        "Pullback": ((C > ema50) & (ema50 > sma200) & (ema50 > ema50.shift(5)) & (L.rolling(3).min() <= ema50 * 1.015)
                     & (C > H.shift(1)) & (R >= 40) & (R <= 65), L.rolling(3).min() * 0.995),
        "Oversold bounce": ((R.shift(1) < 30) & (R > R.shift(1)) & (C > O) & (C > sma200), L * 0.995),
    }
    score = (rs.fillna(0) * 0.30 + (vr.clip(upper=3) / 3).fillna(0) * 20 + (1 - base.clip(upper=0.15) / 0.15).fillna(0) * 15
             + ((ema50 / sma200 - 1).clip(0, 0.2) / 0.2).fillna(0) * 15 + (C / high252).fillna(0) * 20)
    reg = pd.DataFrame(np.repeat(regime.fillna(False).values[:, None], C.shape[1], axis=1), index=C.index, columns=C.columns)
    out = {}
    for name, (raw, struct) in setups.items():
        raw = raw.fillna(False).astype(bool) & liquid.fillna(False)
        if name == "Oversold bounce":  # mean reversion: trend/RS filters don't apply; keep gap rule + regime
            filt = raw & ~gap.fillna(False) & reg
        else:
            filt = raw & quality.fillna(False).astype(bool) & ~avoid.fillna(False) & reg
        out[name] = {"raw": raw, "filtered": filt, "struct": struct}
    return out, score, regime


# ------------------------------------------------------------------ trade simulation
def simulate(sig, struct, px, score, caps, industries):
    """Next-day confirmation entry, then exits. Returns a list of trades."""
    C, O, H, L = (px[k].values for k in ("Close", "Open", "High", "Low"))
    dates, syms = px["Close"].index, px["Close"].columns
    S, ST, SC = sig.values, struct.values, score.values
    trades = []
    for i, j in zip(*np.nonzero(S[:-1])):
        trig = H[i, j] * 1.001
        e = i + 1
        if not (H[e, j] >= trig) or np.isnan(O[e, j]):
            continue  # confirmation failed
        entry = max(O[e, j], trig)
        stop = ST[i, j]
        if np.isnan(stop):
            continue
        risk = (entry - stop) / entry
        if risk > P["max_risk"]:
            continue
        if risk < P["min_risk"]:
            stop, risk = entry * (1 - P["min_risk"]), P["min_risk"]
        target, cur_stop, how, k = entry * (1 + P["target"]), stop, None, e
        for k in range(e, min(e + P["max_days"], len(dates))):
            o, h, l, c = O[k, j], H[k, j], L[k, j], C[k, j]
            if np.isnan(c):
                continue
            if k > e and o <= cur_stop:
                px_out, how = o, "stop" if cur_stop < entry else "breakeven"; break
            if l <= cur_stop:
                px_out, how = cur_stop, "stop" if cur_stop < entry else "breakeven"; break
            if h >= target:
                px_out, how = (o if k > e and o > target else target), "target"; break
            if h >= entry * (1 + P["be_at"]):
                cur_stop = max(cur_stop, entry)
        if how is None:
            if k - e + 1 < P["max_days"] and k == len(dates) - 1:
                continue  # still open at the end of data
            px_out, how = C[k, j], "time"
        ret = px_out / entry - 1 - P["cost"]
        trades.append({"date": dates[e].strftime("%Y-%m-%d"), "exit": dates[k].strftime("%Y-%m-%d"), "symbol": syms[j],
                       "cap": caps.get(syms[j], "?"), "sector": industries.get(syms[j], "Other"), "entry": round(entry, 2),
                       "ret": ret, "r": ret / risk, "risk": risk, "how": how, "days": k - e + 1, "score": float(SC[i, j])})
    return trades


def stats(tr):
    if not tr:
        return {"trades": 0}
    r = np.array([t["ret"] for t in tr]); R = np.array([t["r"] for t in tr])
    wins, losses = r[r > 0], r[r <= 0]
    streak = cur = 0
    for x in sorted(tr, key=lambda t: t["exit"]):
        cur = cur + 1 if x["ret"] <= 0 else 0; streak = max(streak, cur)
    how = pd.Series([t["how"] for t in tr]).value_counts(normalize=True).mul(100).round(1).to_dict()
    return {"trades": len(tr), "win_pct": round(len(wins) / len(r) * 100, 1),
            "avg_win_pct": round(wins.mean() * 100, 2) if len(wins) else 0, "avg_loss_pct": round(losses.mean() * 100, 2) if len(losses) else 0,
            "avg_ret_pct": round(r.mean() * 100, 2), "avg_r": round(R.mean(), 2),
            "profit_factor": round(wins.sum() / -losses.sum(), 2) if losses.sum() < 0 else None,
            "median_days": int(np.median([t["days"] for t in tr])), "worst_streak": streak, "exits_pct": how}


def portfolio(trades):
    """Take trades in date order (best score first) under the risk rules; realised equity curve."""
    eq, open_, curve, taken, paused, month_start = P["capital"], [], [], [], None, {}
    by_day = {}
    for t in trades:
        by_day.setdefault(t["date"], []).append(t)
    days = sorted(set(by_day) | {t["exit"] for t in trades})
    for d in days:
        for t in [x for x in open_ if x["exit"] <= d]:
            eq += t["pnl"]; open_.remove(t); curve.append((t["exit"], round(eq)))
        m = d[:7]
        month_start.setdefault(m, eq)
        if (eq - month_start[m]) / month_start[m] <= P["month_stop"]:
            paused = m
        if paused == m:
            continue
        for t in sorted(by_day.get(d, []), key=lambda x: -x["score"]):
            if len(open_) >= P["max_open"] or sum(o["sector"] == t["sector"] for o in open_) >= P["max_sector"] \
               or any(o["symbol"] == t["symbol"] for o in open_):
                continue
            size = min(eq * P["risk"] / t["risk"], eq * P["pos_cap"], eq - sum(o["size"] for o in open_))
            if size < 5000:
                continue
            open_.append({**t, "size": size, "pnl": size * t["ret"]}); taken.append({**t, "size": round(size), "pnl": round(size * t["ret"])})
    for t in sorted(open_, key=lambda x: x["exit"]):
        eq += t["pnl"]; curve.append((t["exit"], round(eq)))
    if not curve:
        return {"trades": 0}
    s = pd.Series(dict(curve)); peak = s.cummax(); dd = ((s - peak) / peak).min()
    yrs = max((pd.Timestamp(s.index[-1]) - pd.Timestamp(taken[0]["date"])).days / 365.25, 0.1)
    yearly = s.groupby(s.index.str[:4]).last()
    prev = P["capital"]; yr = {}
    for y, v in yearly.items():
        yr[y] = round((v / prev - 1) * 100, 1); prev = v
    return {"start": P["capital"], "end": round(eq), "total_pct": round((eq / P["capital"] - 1) * 100, 1),
            "cagr_pct": round(((eq / P["capital"]) ** (1 / yrs) - 1) * 100, 1), "max_dd_pct": round(dd * 100, 1),
            "trades": len(taken), "win_pct": round(np.mean([t["ret"] > 0 for t in taken]) * 100, 1),
            "by_year_pct": yr, "curve": curve[:: max(1, len(curve) // 300)], "last_trades": taken[-30:]}


# ------------------------------------------------------------------ main
def main():
    uni = load_universe().set_index("symbol")
    tickers = [s + ".NS" for s in uni.index]
    extra = sorted(set(SECTOR_INDEX.values()) | {"^NSEI"})
    data = download(tickers + extra)
    nifty = data["^NSEI"]["Close"]
    idx = nifty.index
    sector_close = {t: data[t]["Close"] for t in extra if t in data}
    stocks = {t[:-3]: d for t, d in data.items() if t.endswith(".NS") and t[:-3] in uni.index}
    print(f"Stocks with enough history: {len(stocks)}")
    px = {f: wide(stocks, f, idx) for f in ("Open", "High", "Low", "Close", "Volume")}
    sigs, score, regime = build_signals(px, nifty, sector_close, uni)
    caps, inds = uni["cap"].to_dict(), uni["industry"].to_dict()

    result = {"generated": datetime.now(IST).strftime("%d %b %Y, %I:%M %p IST"), "params": P,
              "period": [idx[0].strftime("%Y-%m-%d"), idx[-1].strftime("%Y-%m-%d")], "stocks": len(stocks),
              "regime_up_pct": round(regime.mean() * 100, 1), "setups": {}}
    all_filtered = []
    for name, s in sigs.items():
        raw = simulate(s["raw"], s["struct"], px, score, caps, inds)
        flt = simulate(s["filtered"], s["struct"], px, score, caps, inds)
        all_filtered += [{**t, "setup": name} for t in flt]
        by = lambda key: {k: stats([t for t in flt if t[key] == k]) for k in sorted({t[key] for t in flt})}
        result["setups"][name] = {"no_filters": stats(raw), "with_filters": stats(flt), "by_cap": by("cap"),
                                  "by_year": {y: stats([t for t in flt if t["date"][:4] == y]) for y in sorted({t["date"][:4] for t in flt})}}
        print(f"{name}: raw {stats(raw).get('avg_r')}R over {len(raw)} | filtered {stats(flt).get('avg_r')}R over {len(flt)}")
    good = [n for n, v in result["setups"].items() if v["with_filters"].get("trades", 0) >= 30 and v["with_filters"]["avg_r"] > 0.1]
    result["enabled"] = good
    result["portfolio"] = portfolio(sorted([t for t in all_filtered if t["setup"] in good], key=lambda t: t["date"]))
    (ROOT / "backtest.json").write_text(json.dumps(result, default=float, indent=1))
    md = summary(result)
    (ROOT / "backtest_summary.md").write_text(md)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(md)
    print(md)


def summary(r):
    L = [f"# Swing backtest · {r['period'][0]} to {r['period'][1]} · {r['stocks']} stocks\n",
         "| Setup | Filters | Trades | Win % | Avg win | Avg loss | Avg per trade | Worst streak |", "|---|---|---|---|---|---|---|---|"]
    for n, v in r["setups"].items():
        for k, lab in (("no_filters", "none"), ("with_filters", "all")):
            s = v[k]
            if s.get("trades"):
                L.append(f"| {n} | {lab} | {s['trades']} | {s['win_pct']}% | {s['avg_win_pct']}% | {s['avg_loss_pct']}% | {s['avg_r']}R ({s['avg_ret_pct']}%) | {s['worst_streak']} |")
    p = r["portfolio"]
    L.append(f"\n**Enabled setups:** {', '.join(r['enabled']) or 'none'}\n")
    if p.get("trades"):
        L.append(f"**Portfolio (Rs 2.5L, 1% risk, max 4 open):** {p['trades']} trades, win {p['win_pct']}%, "
                 f"total {p['total_pct']}%, CAGR {p['cagr_pct']}%, max drawdown {p['max_dd_pct']}%")
        L.append("\n| Year | Return |\n|---|---|\n" + "\n".join(f"| {y} | {v}% |" for y, v in p["by_year_pct"].items()))
    L.append("\n_Costs 0.3% per trade included. Uses today's index members (survivorship bias: past results look somewhat better than reality). Delivery %, results dates and ASM lists are not in the backtest._\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()
