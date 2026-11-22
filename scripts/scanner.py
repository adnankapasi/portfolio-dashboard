"""Daily swing scanner for Nifty 100 / Midcap 150 / Smallcap 250, with live tracking of every signal.

Called from update.py. Signals are research aids, not tested to make money: the backtest found these
rule types near breakeven in 2024-26, so every signal is tracked forward to measure it on live data.
"""
import io
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

TARGET, BE_AT, MAX_DAYS, COST = 0.075, 0.04, 15, 0.003
ATR_MULT, MIN_RISK, MAX_RISK, MIN_RR = 2.0, 0.02, 0.10, 1.5
MIN_TURNOVER, MIN_PRICE = 5e7, 20
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
           "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9"}


def r2(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 2)


# ------------------------------------------------------------------ data
def load_universe(root):
    p = root / "scripts" / "universe_cache.csv"
    if p.exists():
        return pd.read_csv(p)
    try:
        import backtest
        return backtest.load_universe()
    except SystemExit as e:
        print(f"scanner universe: {e}")
        return None


def download(tickers):
    import yfinance as yf
    out = {}
    for i in range(0, len(tickers), 100):
        chunk = tickers[i:i + 100]
        raw = yf.download(chunk, period="14mo", interval="1d", group_by="ticker", auto_adjust=False, progress=False, threads=True)
        for t in chunk:
            if t in raw.columns.get_level_values(0):
                d = raw[t].dropna(subset=["Close"])
                if len(d) > 60:
                    out[t] = d
    return out


def fetch_delivery(day):
    """NSE full bhavcopy with delivery %. Returns {symbol: deliv_pct} or {}."""
    import requests
    url = f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{day.strftime('%d%m%Y')}.csv"
    try:
        s = requests.Session(); s.headers.update({**HEADERS, "Referer": "https://www.nseindia.com/"})
        r = s.get(url, timeout=30)
        if r.status_code != 200:
            print(f"delivery file: HTTP {r.status_code}"); return {}
        df = pd.read_csv(io.BytesIO(r.content)); df.columns = [c.strip() for c in df.columns]
        df = df[df["SERIES"].astype(str).str.strip() == "EQ"]
        return {str(a).strip(): float(b) for a, b in zip(df["SYMBOL"], pd.to_numeric(df["DELIV_PER"], errors="coerce")) if not np.isnan(b)}
    except Exception as e:
        print(f"delivery file: {e}"); return {}


def fetch_results_calendar():
    """Upcoming board meetings for financial results from NSE. Returns {symbol: 'YYYY-MM-DD'} or None."""
    import requests
    try:
        s = requests.Session(); s.headers.update({**HEADERS, "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-event-calendar"})
        s.get("https://www.nseindia.com/", timeout=15)
        r = s.get("https://www.nseindia.com/api/event-calendar", timeout=30)
        if r.status_code != 200:
            print(f"results calendar: HTTP {r.status_code}"); return None
        out = {}
        for e in r.json():
            if "result" in str(e.get("purpose", "")).lower():
                try:
                    d = datetime.strptime(e["date"], "%d-%b-%Y").strftime("%Y-%m-%d")
                except Exception:
                    continue
                sym = e.get("symbol")
                if sym and (sym not in out or d < out[sym]):
                    out[sym] = d
        print(f"results calendar: {len(out)} companies")
        return out
    except Exception as e:
        print(f"results calendar: {e}"); return None


def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def frames(data, uni, bhav_date, bhav_rows, index):
    """Wide OHLCV frames; today's bar replaced by the official NSE close when available."""
    F = {}
    for f in ("Open", "High", "Low", "Close", "Volume"):
        F[f] = pd.DataFrame({s: data[s + ".NS"][f] for s in uni if s + ".NS" in data})
        F[f].index = pd.to_datetime(F[f].index).tz_localize(None).normalize()
        F[f] = F[f][~F[f].index.duplicated(keep="last")]
    if bhav_date is not None and bhav_rows:
        if bhav_date not in F["Close"].index:
            for f in F:
                F[f].loc[bhav_date] = np.nan
        for s in F["Close"].columns:
            b = bhav_rows.get(s)
            if b:
                F["Open"].loc[bhav_date, s], F["High"].loc[bhav_date, s] = b["open"], b["high"]
                F["Low"].loc[bhav_date, s], F["Close"].loc[bhav_date, s] = b["low"], b["close"]
                if b.get("vol"):
                    F["Volume"].loc[bhav_date, s] = b["vol"]
    for f in F:
        F[f] = F[f].sort_index()
    return F


# ------------------------------------------------------------------ today's scan
def scan(F, nifty, uni, delivery, results, today):
    C, O, H, L, V = F["Close"], F["Open"], F["High"], F["Low"], F["Volume"]
    ema20, ema50 = C.ewm(span=20, adjust=False).mean(), C.ewm(span=50, adjust=False).mean()
    sma200 = C.rolling(200, min_periods=150).mean()
    R = rsi(C)
    pc = C.shift(1)
    atr = np.maximum(np.maximum(H - L, (H - pc).abs()), (L - pc).abs()).ewm(alpha=1 / 14, adjust=False).mean()
    vr = V / V.rolling(20).mean().shift(1)
    turnover = (C * V).rolling(20).mean()
    base = (H.rolling(15).max() - L.rolling(15).min()).shift(1) / C.shift(1)
    W = C.resample("W-FRI").last()
    wup = ((W > W.rolling(30, min_periods=20).mean()) & (W.rolling(10).mean() > W.rolling(30, min_periods=20).mean())).shift(1)
    wup = wup.reindex(C.index, method="ffill").fillna(False)
    n = nifty.reindex(C.index).ffill()
    rel = lambda k: (C / C.shift(k) - 1).sub(n / n.shift(k) - 1, axis=0)
    rs = (rel(63).rank(axis=1, pct=True) + rel(126).rank(axis=1, pct=True)) / 2 * 100
    last = lambda df: df.iloc[-1]
    hh20, hh50 = last(H.rolling(20).max().shift(1)), last(H.rolling(50).max().shift(1))
    hh252 = last(H.rolling(252, min_periods=150).max().shift(1))
    hi52, lo52 = last(H.rolling(252, min_periods=60).max()), last(L.rolling(252, min_periods=60).min())
    c, o, h, cp = last(C), last(O), last(H), C.iloc[-2]
    e20, e50, s200, r_, a_, v_, b_, rs_, w_ = (last(x) for x in (ema20, ema50, sma200, R, atr, vr, base, rs, wup))
    e50_5, r_prev, hprev, low3 = ema50.iloc[-6], R.iloc[-2], H.iloc[-2], L.iloc[-3:].min()
    to = last(turnover)
    score = (rs_.fillna(0) * 0.30 + (v_.clip(upper=3) / 3).fillna(0) * 20 + (1 - b_.clip(upper=0.15) / 0.15).fillna(0) * 15
             + ((e50 / s200 - 1).clip(0, 0.2) / 0.2).fillna(0) * 15 + (c / hi52).fillna(0) * 20)
    date = C.index[-1]
    rows = []
    for s in C.columns:
        if np.isnan(c[s]) or np.isnan(cp[s]):
            continue
        u = uni.loc[s]
        liquid = to[s] >= MIN_TURNOVER and c[s] >= MIN_PRICE
        trend = c[s] > e50[s] > s200[s] if not np.isnan(s200[s]) else c[s] > e50[s]
        setup = None
        if v_[s] >= 1.5 and c[s] > hh252[s]:
            setup = "52-week high"
        elif v_[s] >= 1.5 and c[s] > hh50[s] and trend:
            setup = "Breakout 50d"
        elif v_[s] >= 1.5 and c[s] > hh20[s] and c[s] > e50[s]:
            setup = "Breakout 20d"
        elif trend and e50[s] > e50_5[s] and low3[s] <= e50[s] * 1.015 and c[s] > hprev[s] and 40 <= r_[s] <= 65:
            setup = "Pullback"
        elif r_prev[s] < 30 and r_[s] > r_prev[s] and c[s] > o[s] and c[s] > s200[s]:
            setup = "Oversold bounce"
        watch = setup is None and trend and rs_[s] >= 60 and 0 < (hh20[s] / c[s] - 1) <= 0.03
        trig = round(h[s] * 1.001, 2) if setup else round(hh20[s] * 1.001, 2) if watch else None
        plan = None
        if trig:
            risk = min(max(ATR_MULT * a_[s] / trig, MIN_RISK), 1.0)
            plan = {"trigger": trig, "stop": round(trig * (1 - risk), 2), "target": round(trig * (1 + TARGET), 2),
                    "risk_pct": round(risk * 100, 2), "rr": round(TARGET / risk, 2)}
        res = results.get(s) if results else None
        res_days = (pd.Timestamp(res) - date).days if res else None
        dl = delivery.get(s)
        avoid = []
        if setup:
            if o[s] / cp[s] - 1 > 0.04: avoid.append(f"Gapped up {(o[s] / cp[s] - 1) * 100:.1f}% (chasing)")
            if c[s] / e20[s] - 1 > 0.08 and setup != "Oversold bounce": avoid.append(f"{(c[s] / e20[s] - 1) * 100:.1f}% above 20-day avg (stretched)")
            if r_[s] > 75 and setup != "Oversold bounce": avoid.append(f"RSI {r_[s]:.0f} (overbought)")
            if res_days is not None and 0 <= res_days <= 15: avoid.append(f"Results in {res_days} days")
            if dl is not None and dl < 30 and setup.startswith(("Breakout", "52")): avoid.append(f"Delivery only {dl:.0f}% (speculative volume)")
            if plan and plan["risk_pct"] > MAX_RISK * 100: avoid.append(f"Stop {plan['risk_pct']:.1f}% away (too volatile)")
            elif plan and plan["rr"] < MIN_RR: avoid.append(f"Reward:risk only {plan['rr']}:1 for a 7.5% target")
            if not liquid: avoid.append("Low trading volume")
        if not (setup or watch) and not liquid:
            pass
        checks = {"Weekly trend up": bool(w_[s]), "Strong vs Nifty (top 30%)": bool(rs_[s] >= 70), "Volume 1.5x average": bool(v_[s] >= 1.5),
                  "Tight base before move": bool(b_[s] <= 0.15), "Above 200-day average": bool(c[s] > s200[s]) if not np.isnan(s200[s]) else False,
                  "Near 52-week high (within 10%)": bool(c[s] >= hi52[s] * 0.9)}
        if dl is not None:
            checks["Delivery 40%+"] = bool(dl >= 40)
        rows.append({"symbol": s, "cap": u["cap"], "industry": u["industry"], "close": r2(c[s]), "day_pct": r2((c[s] / cp[s] - 1) * 100),
                     "hi52": r2(hi52[s]), "lo52": r2(lo52[s]), "rsi": r2(r_[s]), "vol_x": r2(v_[s]), "rs": r2(rs_[s]), "deliv": r2(dl),
                     "score": int(round(score[s])), "setup": setup, "watch": bool(watch), "avoid": avoid, "plan": plan, "checks": checks,
                     "results": res, "results_days": res_days, "liquid": bool(liquid),
                     "spark": [r2(x) for x in C[s].iloc[-60:]] if (setup or watch) else None})
    breadth_up = (C.iloc[-1] > C.iloc[-2]).sum(); breadth_dn = (C.iloc[-1] < C.iloc[-2]).sum()
    above50 = float((C.iloc[-1] > ema50.iloc[-1]).mean() * 100)
    ne50, n200 = n.ewm(span=50, adjust=False).mean().iloc[-1], n.rolling(200).mean().iloc[-1]
    nc = n.iloc[-1]
    state = "Uptrend" if nc > ne50 and nc > n200 and above50 >= 50 else "Downtrend" if nc < n200 or above50 < 35 else "Choppy"
    mood = {"state": state, "nifty": r2(nc), "above_ema50": bool(nc > ne50), "above_sma200": bool(nc > n200),
            "pct_above_ema50": round(above50, 1), "up": int(breadth_up), "down": int(breadth_dn)}
    return rows, mood, date


# ------------------------------------------------------------------ live tracking (forward test)
def track(state, F, rows, date, key="scan_log", target=TARGET, be_at=BE_AT, max_days=MAX_DAYS):
    log = state.setdefault(key, [])
    C, O, H, L = F["Close"], F["Open"], F["High"], F["Low"]
    dates = C.index
    for t in log:
        if t["status"] not in ("pending", "open"):
            continue
        s = t["symbol"]
        if s not in C.columns:
            continue
        for d in dates[dates > pd.Timestamp(t["checked"])]:
            o, h, l, c = (x.loc[d, s] for x in (O, H, L, C))
            if np.isnan(c):
                continue
            t["checked"] = d.strftime("%Y-%m-%d")
            if t["status"] == "pending":
                if h >= t["trigger"]:
                    t.update(status="open", entry=r2(max(o, t["trigger"])), entry_date=t["checked"], days=0,
                             cur_stop=r2(max(o, t["trigger"]) * (1 - t["risk_pct"] / 100)))
                    t["target"] = r2(t["entry"] * (1 + target))
                else:
                    t["status"] = "expired"; break
            t["days"] += 1
            first = t["checked"] == t["entry_date"]
            out = None
            if not first and o <= t["cur_stop"]:
                out = (o, "stop" if t["cur_stop"] < t["entry"] else "breakeven")
            elif l <= t["cur_stop"]:
                out = (t["cur_stop"], "stop" if t["cur_stop"] < t["entry"] else "breakeven")
            elif h >= t["target"]:
                out = (o if not first and o > t["target"] else t["target"], "target")
            elif t["days"] >= max_days:
                out = (c, "time")
            if out:
                ret = out[0] / t["entry"] - 1 - COST
                t.update(status="closed", exit=r2(out[0]), exit_date=t["checked"], how=out[1], ret_pct=r2(ret * 100),
                         r=r2(ret / (t["risk_pct"] / 100)))
                break
            if h >= t["entry"] * (1 + be_at):
                t["cur_stop"] = max(t["cur_stop"], t["entry"])
            t["last"] = r2(c)
    active = {t["symbol"] for t in log if t["status"] in ("pending", "open")}
    ds = date.strftime("%Y-%m-%d")
    if not any(t["date"] == ds for t in log):
        for r in rows:
            if r["setup"] and not r["avoid"] and r["plan"] and r["symbol"] not in active:
                log.append({"date": ds, "checked": ds, "symbol": r["symbol"], "cap": r["cap"], "setup": r["setup"], "score": r["score"],
                            "trigger": r["plan"]["trigger"], "risk_pct": r["plan"]["risk_pct"], "status": "pending"})
    state[key] = log[-600:]
    closed = [t for t in log if t["status"] == "closed"]

    def st(ts):
        if not ts:
            return {"trades": 0}
        rr = [t["r"] for t in ts]; rets = [t["ret_pct"] for t in ts]
        return {"trades": len(ts), "win_pct": round(sum(x > 0 for x in rets) / len(ts) * 100, 1), "avg_r": round(float(np.mean(rr)), 2),
                "avg_ret_pct": round(float(np.mean(rets)), 2), "best_pct": max(rets), "worst_pct": min(rets)}
    setups = sorted({t["setup"] for t in log})
    return {"since": log[0]["date"] if log else ds, "overall": st(closed),
            "by_setup": {s: st([t for t in closed if t["setup"] == s]) for s in setups},
            "by_cap": {c: st([t for t in closed if t["cap"] == c]) for c in ("Large", "Mid", "Small")},
            "signals": len(log), "expired": sum(t["status"] == "expired" for t in log),
            "open": [t for t in log if t["status"] == "open"], "pending": [t for t in log if t["status"] == "pending"],
            "recent": sorted(closed, key=lambda t: t["exit_date"], reverse=True)[:25]}


def run(root, state, bhav, nifty_hist=None):
    uni = load_universe(root)
    if uni is None or uni.empty:
        return {"error": "Stock lists not available yet: run the Swing backtest workflow once to download them."}
    uni = uni.drop_duplicates("symbol").set_index("symbol")
    data = download([s + ".NS" for s in uni.index] + ["^NSEI"])
    if "^NSEI" not in data:
        return {"error": "Couldn't download Nifty data today."}
    nifty = data["^NSEI"]["Close"]; nifty.index = pd.to_datetime(nifty.index).tz_localize(None).normalize()
    bdate, brows = (bhav[0], bhav[1]) if bhav else (None, {})
    F = frames(data, uni.index, bdate, brows, nifty.index)
    if bdate is not None and bdate not in nifty.index:
        nifty.loc[bdate] = np.nan; nifty = nifty.sort_index().ffill()
    delivery = fetch_delivery(bdate) if bdate is not None else {}
    results = fetch_results_calendar()
    rows, mood, date = scan(F, nifty, uni, delivery, results, None)
    tracker = track(state, F, rows, date)
    print(f"scanner: {len(rows)} stocks, {sum(1 for r in rows if r['setup'] and not r['avoid'])} setups, "
          f"{sum(1 for r in rows if r['watch'])} watch, {sum(1 for r in rows if r['setup'] and r['avoid'])} avoided")
    return {"date": date.strftime("%Y-%m-%d"), "mood": mood, "rows": rows, "tracker": tracker,
            "params": {"target_pct": TARGET * 100, "atr_mult": ATR_MULT, "max_days": MAX_DAYS, "be_at_pct": BE_AT * 100, "cost_pct": COST * 100},
            "has": {"delivery": bool(delivery), "results": results is not None}}
