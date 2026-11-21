"""Swing-setup backtest for Nifty 100 / Midcap 150 / Smallcap 250 stocks  (round 2).

Round 1 (fixed 4% max stop, +7.5% target, 15 days) lost money on every setup: stops were hit by
normal noise. Round 2 tests a grid of variants and picks them on 2020-2023 only, then checks the
winners on 2024-2026 data they never saw (out-of-sample). Variants:
  entries : 20-day breakout, 50-day breakout, 52-week-high breakout, pullback to 50-EMA, oversold bounce
  filters : basic (liquidity + Nifty above 200-day avg) or full (+ weekly uptrend, RS top 30%, avoid list)
  stops   : 1.5x or 2x the 14-day average true range below entry (max 10%); size keeps Rs risk fixed
  exits   : +7.5% target | 2R target | trailing (close below 20-EMA); breakeven at +1R; time limit
Round-1 notes kept below for reference.


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
SPLIT = "2024-01-01"   # signals before this date choose the rules; after it they only test them
P = dict(target=0.075, max_risk=0.10, min_risk=0.02, be_at=0.04, max_days=25, trail_days=40, cost=0.003,
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
def build(px, nifty):
    C, O, H, L, V = px["Close"], px["Open"], px["High"], px["Low"], px["Volume"]
    ema20, ema50 = C.ewm(span=20, adjust=False).mean(), C.ewm(span=50, adjust=False).mean()
    sma200 = C.rolling(200).mean()
    R = rsi(C)
    pc = C.shift(1)
    tr = np.maximum(np.maximum(H - L, (H - pc).abs()), (L - pc).abs())
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    vr = V / V.rolling(20).mean().shift(1)
    turnover = (C * V).rolling(20).mean()
    base = (H.rolling(15).max() - L.rolling(15).min()).shift(1) / C.shift(1)
    high252 = H.rolling(252, min_periods=120).max()

    W = C.resample("W-FRI").last()
    wup = ((W > W.rolling(30).mean()) & (W.rolling(10).mean() > W.rolling(30).mean())).shift(1)
    wup = wup.reindex(C.index, method="ffill").fillna(False).astype(bool)
    n = nifty.reindex(C.index).ffill()
    rel = lambda k: (C / C.shift(k) - 1).sub(n / n.shift(k) - 1, axis=0)
    rs = (rel(63).rank(axis=1, pct=True) + rel(126).rank(axis=1, pct=True)) / 2 * 100

    full = lambda f: pd.DataFrame(np.repeat(f.fillna(False).values[:, None], C.shape[1], axis=1), index=C.index, columns=C.columns)
    regime = full(n > n.rolling(200).mean())
    liquid = ((turnover >= P["min_turnover"]) & (C >= P["min_price"])).fillna(False)
    gap = (O / C.shift(1) - 1 > 0.04).fillna(False)
    avoid = gap | (C / ema20 - 1 > 0.08).fillna(False) | (R > 75).fillna(False)
    quality = wup & (rs >= P["rs_min"]).fillna(False)
    trend = (C > ema50) & (ema50 > sma200)

    entries = {
        "Breakout 20d": (C > H.rolling(20).max().shift(1)) & (vr >= 1.5) & (C > ema50),
        "Breakout 50d": (C > H.rolling(50).max().shift(1)) & (vr >= 1.5) & trend,
        "52-week high": (C > H.rolling(252, min_periods=200).max().shift(1)) & (vr >= 1.5),
        "Pullback": trend & (ema50 > ema50.shift(5)) & (L.rolling(3).min() <= ema50 * 1.015) & (C > H.shift(1)) & (R >= 40) & (R <= 65),
        "Oversold bounce": (R.shift(1) < 30) & (R > R.shift(1)) & (C > O) & (C > sma200),
    }
    score = (rs.fillna(0) * 0.30 + (vr.clip(upper=3) / 3).fillna(0) * 20 + (1 - base.clip(upper=0.15) / 0.15).fillna(0) * 15
             + ((ema50 / sma200 - 1).clip(0, 0.2) / 0.2).fillna(0) * 15 + (C / high252).fillna(0) * 20)
    sigs = {}
    for name, e in entries.items():
        e = e.fillna(False).astype(bool) & liquid & regime
        sigs[(name, "basic")] = e & ~gap
        sigs[(name, "full")] = e & ~gap if name == "Oversold bounce" else e & quality & ~avoid
    return sigs, score, atr, ema20


# ------------------------------------------------------------------ trade simulation
def simulate(sig, atr, k_atr, exit_mode, px, ema20, score, caps, industries):
    """Next-day confirmation entry (above signal-day high), ATR stop, chosen exit."""
    C, O, H, L = (px[f].values for f in ("Close", "Open", "High", "Low"))
    A, E20, SC = atr.values, ema20.values, score.values
    dates, syms = px["Close"].index, px["Close"].columns
    S = sig.values
    max_days = P["trail_days"] if exit_mode == "trail" else P["max_days"]
    trades = []
    for i, j in zip(*np.nonzero(S[:-1])):
        trig, e = H[i, j] * 1.001, i + 1
        if np.isnan(O[e, j]) or not (H[e, j] >= trig) or np.isnan(A[i, j]):
            continue
        entry = max(O[e, j], trig)
        risk = k_atr * A[i, j] / entry
        if risk > P["max_risk"]:
            continue
        risk = max(risk, P["min_risk"])
        stop = entry * (1 - risk)
        target = entry * (1 + P["target"]) if exit_mode == "t75" else entry * (1 + 2 * risk) if exit_mode == "2r" else np.inf
        be_level = entry * (1 + risk)  # breakeven after +1R
        cur, how, k = stop, None, e
        for k in range(e, min(e + max_days, len(dates))):
            o, h, l, c = O[k, j], H[k, j], L[k, j], C[k, j]
            if np.isnan(c):
                continue
            if k > e and o <= cur:
                px_out, how = o, "stop" if cur < entry else "breakeven"; break
            if l <= cur:
                px_out, how = cur, "stop" if cur < entry else "breakeven"; break
            if h >= target:
                px_out, how = (o if k > e and o > target else target), "target"; break
            if exit_mode == "trail" and k > e and c < E20[k, j]:
                px_out, how = c, "trail"; break
            if h >= be_level:
                cur = max(cur, entry)
        if how is None:
            if k == len(dates) - 1 and k - e + 1 < max_days:
                continue
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
EXITS = {"t75": "+7.5% target", "2r": "2R target", "trail": "Trailing 20-EMA"}


def main():
    uni = load_universe().set_index("symbol")
    tickers = [s + ".NS" for s in uni.index]
    data = download(tickers + ["^NSEI"])
    nifty = data["^NSEI"]["Close"]
    idx = nifty.index
    stocks = {t[:-3]: d for t, d in data.items() if t.endswith(".NS") and t[:-3] in uni.index}
    print(f"Stocks with enough history: {len(stocks)}")
    px = {f: wide(stocks, f, idx) for f in ("Open", "High", "Low", "Close", "Volume")}
    sigs, score, atr, ema20 = build(px, nifty)
    caps, inds = uni["cap"].to_dict(), uni["industry"].to_dict()

    variants = []
    for (entry, filt), sig in sigs.items():
        for k_atr in (1.5, 2.0):
            for ex in EXITS:
                tr = simulate(sig, atr, k_atr, ex, px, ema20, score, caps, inds)
                ins, oos = [t for t in tr if t["date"] < SPLIT], [t for t in tr if t["date"] >= SPLIT]
                v = {"name": f"{entry} · {filt} filters · {k_atr}x ATR stop · {EXITS[ex]}", "entry": entry, "filters": filt,
                     "atr": k_atr, "exit": ex, "in_sample": stats(ins), "out_of_sample": stats(oos), "all": stats(tr),
                     "by_cap_oos": {c: stats([t for t in oos if t["cap"] == c]) for c in ("Large", "Mid", "Small")}, "_trades": tr}
                variants.append(v)
                print(f"{v['name']:<72} in {v['in_sample'].get('avg_r', '-')}R/{len(ins)}  out {v['out_of_sample'].get('avg_r', '-')}R/{len(oos)}")

    # choose on in-sample only: enough trades and positive edge, ranked by edge x sqrt(trades)
    cand = [v for v in variants if v["in_sample"].get("trades", 0) >= 30 and v["in_sample"]["avg_r"] > 0.1]
    cand.sort(key=lambda v: -v["in_sample"]["avg_r"] * np.sqrt(v["in_sample"]["trades"]))
    picked = cand[:5]
    passed = [v for v in picked if v["out_of_sample"].get("trades", 0) >= 20 and v["out_of_sample"]["avg_r"] > 0.1]

    result = {"generated": datetime.now(IST).strftime("%d %b %Y, %I:%M %p IST"), "params": P, "split": SPLIT,
              "period": [idx[0].strftime("%Y-%m-%d"), idx[-1].strftime("%Y-%m-%d")], "stocks": len(stocks),
              "variants_tested": len(variants), "picked_in_sample": [v["name"] for v in picked], "passed": [v["name"] for v in passed]}
    if passed:
        best = passed[0]
        result["best"] = {k: v for k, v in best.items() if k != "_trades"}
        result["portfolio_all"] = portfolio(sorted(best["_trades"], key=lambda t: t["date"]))
        result["portfolio_oos"] = portfolio(sorted([t for t in best["_trades"] if t["date"] >= SPLIT], key=lambda t: t["date"]))
    result["variants"] = sorted([{k: v for k, v in x.items() if k != "_trades"} for x in variants],
                                key=lambda v: -(v["in_sample"].get("avg_r") or -9))
    (ROOT / "backtest.json").write_text(json.dumps(result, default=float, indent=1))
    md = summary(result)
    (ROOT / "backtest_summary.md").write_text(md)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(md)
    print(md)


def summary(r):
    f = lambda s: f"{s['trades']} · {s['win_pct']}% · {s['avg_r']}R" if s.get("trades") else "–"
    L = [f"# Swing backtest round 2 · {r['period'][0]} to {r['period'][1]} · {r['stocks']} stocks · {r['variants_tested']} variants\n",
         f"Rules chosen on signals before {r['split']}; checked on {r['split']} onwards (unseen).\n",
         "## Top 15 variants by 2020–23 results\n", "| Variant | 2020–23 (trades · win · avg) | 2024–26 unseen | Exits 2024–26 |", "|---|---|---|---|"]
    for v in r["variants"][:15]:
        ex = ", ".join(f"{k} {x}%" for k, x in v["out_of_sample"].get("exits_pct", {}).items())
        L.append(f"| {v['name']} | {f(v['in_sample'])} | {f(v['out_of_sample'])} | {ex} |")
    L.append(f"\n**Picked on 2020–23:** {len(r['picked_in_sample'])} · **Passed on unseen 2024–26:** {', '.join(r['passed']) or 'none'}\n")
    for key, lab in (("portfolio_all", "Portfolio 2020–26"), ("portfolio_oos", "Portfolio 2024–26 only (unseen)")):
        p = r.get(key)
        if p and p.get("trades"):
            L.append(f"**{lab} (Rs 2.5L, 1% risk, max 4 open):** {p['trades']} trades, win {p['win_pct']}%, total {p['total_pct']}%, "
                     f"CAGR {p['cagr_pct']}%, max drawdown {p['max_dd_pct']}% · by year: " + ", ".join(f"{y} {x}%" for y, x in p["by_year_pct"].items()))
    L.append("\n_Costs 0.3% per trade included. Today's index members (survivorship bias). Delivery %, results dates and ASM lists not tested._\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()
