"""ETF themes: signals for investing (1-3 and 6-12 months), rotation ranking, best ETF per theme,
and swing setups on ETFs (+5% target, 2.5% stop) tracked forward like the stock scanner.
Weights are set from reasoning, not fitted. Debt ETFs are excluded on purpose.
"""
import re
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import numpy as np
import pandas as pd

TGT, STOP, BE_AT, MAX_DAYS, COST = 0.05, 0.025, 0.025, 20, 0.002
MIN_TURNOVER = 2e7  # Rs 2 crore a day
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}

# name, category, price-history sources to try (index first), currency, known ETF symbols, name keywords, driver, news query
THEMES = [
    ("Nifty 50", "Broad", ["^NSEI"], "INR", ["NIFTYBEES", "SETFNIF50", "NIFTYIETF", "HDFCNIFTY", "UTINIFTETF", "NIF100BEES"], [r"NIFTY ?50 ?(ETF|BEES)", r"NIFTY ?BEES"], "vix", "Nifty 50"),
    ("Nifty Next 50", "Broad", ["^NSMIDCP"], "INR", ["JUNIORBEES", "SETFNN50", "NEXT50IETF", "NEXT50"], [r"NEXT ?50", r"JUNIOR"], "vix", "Nifty Next 50"),
    ("Midcap 150", "Broad", ["MID150BEES.NS", "MIDCAPETF.NS"], "INR", ["MID150BEES", "MIDCAPETF", "MID150", "MIDCAPIETF"], [r"MIDCAP ?150"], "vix", "midcap stocks India"),
    ("Smallcap 250", "Broad", ["HDFCSML250.NS", "SMALLCAP.NS"], "INR", ["HDFCSML250", "SMALLCAP", "SMALL250"], [r"SMALLCAP ?250", r"SMALL ?CAP"], "vix", "smallcap stocks India"),
    ("Momentum 30", "Smart beta", ["MOM30IETF.NS", "MOMENTUM.NS"], "INR", ["MOM30IETF", "MOMENTUM", "MOMOMENTUM", "MOM100"], [r"MOMENTUM ?30", r"MOMENTUM"], "vix", "momentum stocks India"),
    ("Low volatility 30", "Smart beta", ["LOWVOLIETF.NS", "LOWVOL1.NS"], "INR", ["LOWVOLIETF", "LOWVOL1", "LOWVOL"], [r"LOW ?VOL"], "vix", "low volatility stocks India"),
    ("Bank", "Sectors", ["^NSEBANK"], "INR", ["BANKBEES", "SETFNIFBK", "BANKIETF", "BANKETF"], [r"BANK ?(BEES|ETF)", r"NIFTY ?BANK"], None, "Nifty Bank"),
    ("PSU Bank", "Sectors", ["^CNXPSUBANK"], "INR", ["PSUBNKBEES", "PSUBANK", "PSUBNKIETF"], [r"PSU ?BANK"], None, "PSU bank stocks"),
    ("IT", "Sectors", ["^CNXIT"], "INR", ["ITBEES", "ITIETF", "IT", "ITETF", "TECH"], [r"\bIT ?(BEES|ETF)", r"NIFTY ?IT"], "it", "Indian IT stocks"),
    ("Pharma", "Sectors", ["^CNXPHARMA"], "INR", ["PHARMABEES", "PHARMAIETF", "HEALTHIETF", "HEALTHY"], [r"PHARMA", r"HEALTH"], "pharma", "Indian pharma stocks"),
    ("Auto", "Sectors", ["^CNXAUTO"], "INR", ["AUTOBEES", "AUTOIETF", "AUTOETF"], [r"\bAUTO"], None, "Indian auto stocks"),
    ("FMCG", "Sectors", ["^CNXFMCG"], "INR", ["FMCGIETF", "FMCGETF", "CONSUMBEES", "CONSUMIETF"], [r"FMCG", r"CONSUMPTION"], None, "FMCG stocks India"),
    ("Metal", "Sectors", ["^CNXMETAL"], "INR", ["METALIETF", "METAL", "METALETF"], [r"METAL"], "metal", "metal stocks India"),
    ("Energy", "Sectors", ["^CNXENERGY"], "INR", ["MOENERGY", "ENERGY", "ENERGYETF", "OILIETF"], [r"ENERGY", r"\bOIL"], "energy", "energy stocks India"),
    ("Infra", "Sectors", ["^CNXINFRA"], "INR", ["INFRABEES", "INFRAIETF", "INFRA"], [r"INFRA"], None, "infrastructure stocks India"),
    ("Nasdaq 100", "International", ["^NDX"], "USD", ["MON100", "N100", "MAFANG", "MONQ50"], [r"NASDAQ"], "intl", "Nasdaq 100"),
    ("S&P 500", "International", ["^GSPC"], "USD", ["MASPTOP50", "SPXTOP50", "MOSP500"], [r"S ?& ?P ?500", r"S&P"], "intl", "S&P 500"),
    ("Hang Seng", "International", ["^HSI"], "HKD", ["HNGSNGBEES", "MAHKTECH", "HANGSENG"], [r"HANG ?SENG"], "intl", "Hang Seng"),
    ("Gold", "Gold & silver", ["GC=F"], "USD", ["GOLDBEES", "GOLDIETF", "SETFGOLD", "HDFCGOLD", "GOLD1", "AXISGOLD"], [r"\bGOLD"], "metals", None),
    ("Silver", "Gold & silver", ["SI=F"], "USD", ["SILVERBEES", "SILVERIETF", "SILVER", "HDFCSILVER", "SILVER1"], [r"SILVER"], "metals", None),
]
EXTRA = {"usdinr": "INR=X", "copper": "HG=F", "crude": "CL=F", "nasdaq": "^NDX", "ivix": "^INDIAVIX"}
W13 = {"trend": .35, "mom3": .25, "rs": .25, "drv": .15}
W612 = {"mom12": .35, "rs": .30, "trend": .20, "drv": .15}


def r2(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 2)


def label(x):
    return "No data" if x is None or np.isnan(x) else "Positive" if x > 0.25 else "Negative" if x < -0.25 else "Neutral"


def sgn(x, thr):
    return pd.Series(np.where(x > thr, 1.0, np.where(x < -thr, -1.0, 0.0)), index=x.index).where(x.notna())


# ------------------------------------------------------------------ data
def download(tickers, period="max"):
    import yfinance as yf
    out = {}
    tickers = sorted(set(tickers))
    for i in range(0, len(tickers), 60):
        raw = yf.download(tickers[i:i + 60], period=period, interval="1d", group_by="ticker", auto_adjust=False, progress=False, threads=True)
        for t in tickers[i:i + 60]:
            if t in raw.columns.get_level_values(0):
                d = raw[t].dropna(subset=["Close"])
                d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
                d = d[~d.index.duplicated(keep="last")]
                if len(d) > 60:
                    out[t] = d
    return out


def pick_etfs(bhav_raw):
    """For each theme, ETFs found in today's NSE file (known symbols + name keywords), most traded first."""
    if bhav_raw is None:
        return {}
    df = bhav_raw.copy()
    name = df.get("FinInstrmNm", pd.Series("", index=df.index)).astype(str).str.upper()
    etfish = name.str.contains("ETF|BEES|EXCHANGE TRADED", regex=True) | df["TckrSymb"].astype(str).str.contains("BEES|IETF|ETF")
    turn = pd.to_numeric(df.get("TtlTrfVal", 0), errors="coerce").fillna(0)
    out = {}
    for th in THEMES:
        known = df["TckrSymb"].isin(th[4])
        kw = etfish & name.apply(lambda n: any(re.search(k, n) for k in th[5]))
        cand = df[known | kw].assign(turn=turn).sort_values("turn", ascending=False).drop_duplicates("TckrSymb")
        out[th[0]] = [{"symbol": r.TckrSymb, "turnover_cr": round(float(r.turn) / 1e7, 2), "close": float(r.ClsPric),
                       "name": str(getattr(r, "FinInstrmNm", "")).title()} for r in cand.itertuples()][:4]
    return out


def news_for(query):
    import requests
    try:
        xml = requests.get("https://news.google.com/rss/search", headers=HEADERS, timeout=20,
                           params={"q": query + " when:7d", "hl": "en-IN", "gl": "IN", "ceid": "IN:en"}).content
        items = []
        for it in list(ET.fromstring(xml).iter("item"))[:3]:
            try:
                d = parsedate_to_datetime(it.findtext("pubDate")).strftime("%Y-%m-%d")
            except Exception:
                d = ""
            items.append({"title": (it.findtext("title") or "").strip(), "link": it.findtext("link"), "date": d})
        return items
    except Exception as e:
        print(f"ETF news '{query}': {e}")
        return []


# ------------------------------------------------------------------ signals
def series_for(th, data, fx):
    """Theme price history in rupees. Returns (series, source ticker)."""
    for t in th[2]:
        if t in data and len(data[t]) > 300:
            s = data[t]["Close"]
            if th[3] == "USD" and fx is not None:
                s = s * fx.reindex(s.index).ffill()
            elif th[3] == "HKD" and fx is not None:
                s = s * fx.reindex(s.index).ffill() / 7.8
            return s.dropna(), t
    return None, None


def driver(kind, idx, X):
    """Theme-specific driver: +1 helps, -1 hurts, 0 neutral, NaN unavailable."""
    nan = pd.Series(np.nan, index=idx)
    g = lambda k: X[k].reindex(idx).ffill() if k in X else None
    u, cu, cr, nq, iv = g("usdinr"), g("copper"), g("crude"), g("nasdaq"), g("ivix")
    if kind in ("it", "pharma", "intl") and u is not None:
        d = sgn(u / u.shift(63) - 1, 0.01)  # weaker rupee helps exporters and foreign assets
        if kind == "it" and nq is not None:
            d = ((d + pd.Series(np.where(nq > nq.rolling(200).mean(), 1.0, -1.0), index=idx)) / 2).round()
        return d
    if kind == "metal" and cu is not None:
        return sgn(cu / cu.shift(63) - 1, 0.05)
    if kind == "energy" and cr is not None:
        return sgn(cr / cr.shift(63) - 1, 0.10)
    if kind == "vix" and iv is not None:
        return pd.Series(np.where(iv > 22, -1.0, np.where(iv < 13, 1.0, 0.0)), index=idx).where(iv.notna())
    return nan


def build(P, X):
    """P: DataFrame dates x themes (rupee prices). Returns factor dict of DataFrames."""
    ma50, ma200 = P.rolling(50).mean(), P.rolling(200).mean()
    trend = pd.DataFrame(np.where((P > ma50) & (P > ma200), 1.0, np.where((P < ma50) & (P < ma200), -1.0, 0.0)), index=P.index, columns=P.columns).where(ma200.notna())
    mom3 = (P / P.shift(63) - 1).apply(lambda c: sgn(c, 0.02))
    mom12 = (P.shift(21) / P.shift(252) - 1).apply(lambda c: sgn(c, 0.03))
    rel = ((P / P.shift(63)).rank(axis=1, pct=True) + (P / P.shift(126)).rank(axis=1, pct=True)) / 2
    rs = pd.DataFrame(np.where(rel >= 2 / 3, 1.0, np.where(rel <= 1 / 3, -1.0, 0.0)), index=P.index, columns=P.columns).where(rel.notna())
    drv = pd.DataFrame({th[0]: driver(th[6], P.index, X) for th in THEMES if th[0] in P.columns})
    return {"trend": trend, "mom3": mom3, "mom12": mom12, "rs": rs, "drv": drv}, rel


def combine(F, W):
    num = sum(F[k].fillna(0) * w for k, w in W.items())
    den = sum(F[k].notna() * w for k, w in W.items())
    return (num / den).where(den >= 0.6)


def history_test(P, c13, c612, nifty):
    """Pooled forward returns by label, and a monthly top-3 rotation vs Nifty 50."""
    out = {}
    me = P.groupby(P.index.to_period("M")).tail(1).index
    for name, comp, days in (("1–3 months", c13, 63), ("6–12 months", c612, 252)):
        fwd = (P.shift(-days) / P - 1).loc[me]
        lab = comp.loc[me].apply(lambda col: col.map(label))
        rows = pd.DataFrame({"f": fwd.stack(), "l": lab.stack()}).dropna()
        rows = rows[rows.l != "No data"]
        if len(rows) < 100:
            continue
        res = {"from": str(me[0].year), "to": str(me[-1].year), "obs": len(rows), "all": {"avg_pct": round(rows.f.mean() * 100, 1)}}
        for l in ("Positive", "Neutral", "Negative"):
            g = rows[rows.l == l].f
            res[l] = {"n": int(len(g)), "avg_pct": round(g.mean() * 100, 1) if len(g) else None, "up_pct": round((g > 0).mean() * 100) if len(g) else None}
        out[name] = res
    # rotation: at each month end hold the 3 strongest themes (by relative strength) for the next month
    eq = [t for t in P.columns if t not in ("Gold", "Silver")]
    m = P[eq].loc[me]
    rel = ((m / m.shift(3)).rank(axis=1, pct=True) + (m / m.shift(6)).rank(axis=1, pct=True)) / 2
    nxt = m.shift(-1) / m - 1
    rets, prev = [], set()
    for d in rel.index[6:-1]:
        top = set(rel.loc[d].dropna().nlargest(3).index)
        if len(top) < 3:
            continue
        cost = len(top - prev) / 3 * 2 * COST
        rets.append((d, nxt.loc[d, list(top)].mean() - cost)); prev = top
    if len(rets) >= 36 and nifty is not None:
        r = pd.Series(dict(rets))
        n = (nifty.reindex(me).shift(-1) / nifty.reindex(me) - 1).reindex(r.index)
        yrs = len(r) / 12
        cagr = lambda x: ((1 + x).prod() ** (1 / yrs) - 1) * 100
        curve = (1 + r).cumprod(); dd = (curve / curve.cummax() - 1).min()
        out["rotation"] = {"from": str(r.index[0].year), "years": round(yrs, 1), "cagr_pct": round(cagr(r), 1), "nifty_cagr_pct": round(cagr(n.fillna(0)), 1),
                           "max_dd_pct": round(dd * 100, 1), "worst_month_pct": round(r.min() * 100, 1), "beat_pct": round((r > n).mean() * 100)}
    return out


# ------------------------------------------------------------------ swing setups on ETFs
def merge_today(data, chosen, bhav_rows, bdate):
    """Use NSE's official bar for today on each chosen ETF."""
    if bdate is None:
        return
    for e in chosen.values():
        t, b = e["symbol"] + ".NS", (bhav_rows or {}).get(e["symbol"])
        if t not in data or not b:
            continue
        d = data[t]
        if bdate not in d.index:
            d.loc[bdate] = np.nan
        d.loc[bdate, ["Open", "High", "Low", "Close"]] = [b["open"], b["high"], b["low"], b["close"]]
        if b.get("vol"):
            d.loc[bdate, "Volume"] = b["vol"]
        data[t] = d.sort_index()


def etf_swing(data, chosen, sig612, prem):
    rows = []
    for theme, e in chosen.items():
        d = data.get(e["symbol"] + ".NS")
        if d is None or len(d) < 60:
            continue
        C, O, H, L, V = d["Close"], d["Open"], d["High"], d["Low"], d["Volume"]
        ema50, sma200 = C.ewm(span=50, adjust=False).mean(), C.rolling(200, min_periods=120).mean()
        pc = C.shift(1)
        atr = np.maximum(np.maximum(H - L, (H - pc).abs()), (L - pc).abs()).ewm(alpha=1 / 14, adjust=False).mean()
        vr = V / V.rolling(20).mean().shift(1)
        c, h = C.iloc[-1], H.iloc[-1]
        trend = c > ema50.iloc[-1] > (sma200.iloc[-1] if not np.isnan(sma200.iloc[-1]) else 0)
        setup = None
        if c > H.rolling(252, min_periods=150).max().shift(1).iloc[-1] and vr.iloc[-1] >= 1.3:
            setup = "52-week high"
        elif c > H.rolling(50).max().shift(1).iloc[-1] and vr.iloc[-1] >= 1.3 and trend:
            setup = "Breakout 50d"
        elif c > H.rolling(20).max().shift(1).iloc[-1] and vr.iloc[-1] >= 1.3 and c > ema50.iloc[-1]:
            setup = "Breakout 20d"
        elif trend and L.iloc[-3:].min() <= ema50.iloc[-1] * 1.01 and c > H.iloc[-2]:
            setup = "Pullback"
        if not setup:
            continue
        avoid = []
        if sig612.get(theme) == "Negative": avoid.append("Theme 6–12 m signal is Negative")
        if prem.get(theme) is not None and prem[theme] > 2: avoid.append(f"Trades about {prem[theme]:.1f}% above fair value")
        if e["turnover_cr"] * 1e7 < MIN_TURNOVER: avoid.append("Low trading volume")
        if 1.5 * atr.iloc[-1] / c > STOP: avoid.append(f"Too volatile for a {STOP * 100:.1f}% stop")
        trig = round(h * 1.001, 2)
        score = int(round(min(vr.iloc[-1], 3) / 3 * 30 + (40 if trend else 15) + (30 if sig612.get(theme) == "Positive" else 15 if sig612.get(theme) == "Neutral" else 0)))
        rows.append({"symbol": e["symbol"], "theme": theme, "cap": "ETF", "setup": setup, "score": score, "avoid": avoid,
                     "close": r2(c), "day_pct": r2((c / C.iloc[-2] - 1) * 100), "vol_x": r2(vr.iloc[-1]),
                     "plan": {"trigger": trig, "stop": round(trig * (1 - STOP), 2), "target": round(trig * (1 + TGT), 2), "risk_pct": STOP * 100, "rr": TGT / STOP},
                     "atr_pct": r2(atr.iloc[-1] / c * 100)})
    return rows


def run(state, bhav, metals_out=None, holdings=None):
    bdate, brows, raw = (bhav[0], bhav[1], bhav[3]) if bhav else (None, {}, None)
    picks = pick_etfs(raw)
    chosen = {t: v[0] for t, v in picks.items() if v}
    tick = [s for th in THEMES for s in th[2]] + list(EXTRA.values()) + [e["symbol"] + ".NS" for e in chosen.values()]
    data = download(tick)
    merge_today(data, chosen, brows, bdate)
    X = {k: data[t]["Close"] for k, t in EXTRA.items() if t in data}
    fx = X.get("usdinr")
    series, src = {}, {}
    for th in THEMES:
        s, t = series_for(th, data, fx)
        if s is not None:
            series[th[0]], src[th[0]] = s, t
    if "Nifty 50" not in series:
        return {"error": "Couldn't download index data for ETF themes today."}
    P = pd.DataFrame(series).sort_index()
    P = P.reindex(series["Nifty 50"].index[series["Nifty 50"].index >= "2005-01-01"]).ffill(limit=5)
    F, rel = build(P, X)
    c13, c612 = combine(F, W13), combine(F, W612)
    hist = history_test(P, c13, c612, P["Nifty 50"])

    # premium vs fair value (estimated): ETF price relative to its underlying, vs that ratio's normal level
    prem = {}
    for th in THEMES:
        e = chosen.get(th[0]); und = series.get(th[0])
        d = data.get(e["symbol"] + ".NS") if e else None
        if d is None or und is None or not src.get(th[0], "").startswith(("^", "GC", "SI")):
            continue
        ratio = (d["Close"] / und.reindex(d.index).ffill()).dropna()
        if len(ratio) > 120:
            prem[th[0]] = round(float((ratio.iloc[-1] / ratio.iloc[-250:].median() - 1) * 100), 1)

    last = P.index[-1]
    rank = rel.loc[last].rank(ascending=False, method="min")
    MET = {}
    if metals_out and not metals_out.get("error"):
        for m in ("gold", "silver"):
            hz = {h["horizon"]: h["label"] for h in metals_out[m]["horizons"]}
            MET["Gold" if m == "gold" else "Silver"] = (hz.get("1–3 months"), hz.get("6–12 months"))
    DN = {"trend": "Price trend", "mom3": "3-month momentum", "mom12": "12-month momentum", "rs": "Strength vs other themes"}
    DRV_NAME = {"it": "Rupee and Nasdaq trend", "pharma": "Rupee (exports)", "intl": "Rupee", "metal": "Copper price (global metals)",
                "energy": "Crude oil", "vix": "India VIX (fear)"}
    themes = []
    for th in THEMES:
        n = th[0]
        if n not in P.columns or np.isnan(P[n].iloc[-1]):
            continue
        p = P[n].dropna()
        ret = lambda k: r2((p.iloc[-1] / p.iloc[-1 - k] - 1) * 100) if len(p) > k else None
        s13, s612 = label(c13[n].iloc[-1]), label(c612[n].iloc[-1])
        if n in MET:
            s13, s612 = MET[n]
        drivers = []
        for k, nm in DN.items():
            v = F[k][n].iloc[-1]
            if not np.isnan(v):
                drivers.append({"name": nm, "effect": {1.0: "help", -1.0: "hurt"}.get(v, "neutral")})
        if th[6] in DRV_NAME and n in F["drv"] and not np.isnan(F["drv"][n].iloc[-1]):
            drivers.append({"name": DRV_NAME[th[6]], "effect": {1.0: "help", -1.0: "hurt"}.get(F["drv"][n].iloc[-1], "neutral")})
        if prem.get(n) is not None and prem[n] > 2:
            drivers.append({"name": "Premium over fair value", "effect": "hurt", "value": f"About {prem[n]:.1f}% above its usual level"})
        themes.append({"name": n, "cat": th[1], "s13": s13, "s612": s612, "score13": r2(c13[n].iloc[-1]), "score612": r2(c612[n].iloc[-1]),
                       "rank": int(rank[n]) if not pd.isna(rank.get(n)) else None, "m1": ret(21), "m3": ret(63), "y1": ret(252),
                       "etf": chosen.get(n), "alts": picks.get(n, [])[1:], "premium": prem.get(n), "drivers": drivers, "source": src.get(n),
                       "news": news_for(th[7]) if th[7] else []})
    themes.sort(key=lambda t: t["rank"] or 99)
    sig612 = {t["name"]: t["s612"] for t in themes}
    swing = etf_swing(data, chosen, sig612, prem)
    # forward-track ETF setups with their own rules, reusing the stock scanner's tracker
    tracker = None
    try:
        import scanner
        Fr = {f: pd.DataFrame({e["symbol"]: data[e["symbol"] + ".NS"][f] for e in chosen.values() if e["symbol"] + ".NS" in data}) for f in ("Open", "High", "Low", "Close")}
        if len(Fr["Close"].columns):
            tracker = scanner.track(state, Fr, swing, Fr["Close"].index[-1], key="etf_log", target=TGT, be_at=BE_AT, max_days=MAX_DAYS)
    except Exception as e:
        print(f"ETF tracker: {e}")
    held = []
    for h in holdings or []:
        for t in themes:
            syms = {x["symbol"] for x in ([t["etf"]] if t["etf"] else []) + t["alts"]} | set(next(th[4] for th in THEMES if th[0] == t["name"]))
            if h["symbol"] in syms:
                held.append({"symbol": h["symbol"], "broker": h.get("broker", "Kite"), "theme": t["name"], "pnl_pct": r2(h["pnl_pct"]), "s612": t["s612"], "s13": t["s13"]})
    lead = [t["name"] for t in themes if t["rank"]][:3]
    lag = [t["name"] for t in themes if t["rank"]][-3:][::-1]
    print(f"ETFs: {len(themes)} themes, {len(chosen)} ETFs picked, {sum(1 for r in swing if not r['avoid'])} swing setups")
    return {"date": last.strftime("%Y-%m-%d"), "themes": themes, "leading": lead, "lagging": lag, "history": hist,
            "swing": swing, "tracker": tracker, "held": held,
            "rules": {"target_pct": TGT * 100, "stop_pct": STOP * 100, "be_at_pct": BE_AT * 100, "max_days": MAX_DAYS}}
