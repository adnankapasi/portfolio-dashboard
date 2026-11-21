"""Gold and silver outlook: tracks the main price drivers, scores each as helping / hurting,
combines them into outlooks for 4 horizons, and tests the 1-3 month and 6-12 month outlooks
on 20 years of history. Weights are set from economic reasoning, not fitted to the data.

Sources (all free, no keys): Yahoo (prices, dollar index, USD/INR, copper, VIX),
FRED (US 10y real yield, fed funds rate, CPI), CFTC (futures positioning),
SPDR (GLD gold holdings), Google News RSS (headlines).
"""
import io
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime

import numpy as np
import pandas as pd

OZ = 31.1035
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
YAHOO = {"gold": "GC=F", "silver": "SI=F", "dxy": "DX-Y.NYB", "usdinr": "INR=X", "copper": "HG=F", "vix": "^VIX"}
FRED = {"ry": "DFII10", "ff": "DFF", "cpi": "CPIAUCSL"}
COT_CODES = {"cot_g": "088691", "cot_s": "084691"}
NEWS_Q = ["gold price", "silver price", "gold import duty India", "Federal Reserve interest rates", "central bank gold buying"]
TAGS = {"Fed": ["fed ", "federal reserve", "rate hike", "rate cut", "powell", "fomc"], "Policy": ["import duty", "customs duty", "gst", "budget", "rbi"],
        "Central banks": ["central bank", "pboc", "reserve bank"], "ETF flows": ["etf", "inflow", "outflow"],
        "Geopolitics": ["war", "tension", "sanction", "conflict", "tariff"], "Dollar": ["dollar", "usd"], "Industry": ["solar", "industrial", "demand"]}

# ------------------------------------------------------------------ fetching (each source optional)
def get(url, **kw):
    import requests
    r = requests.get(url, headers=HEADERS, timeout=30, **kw)
    r.raise_for_status()
    return r


def fetch_yahoo():
    import yfinance as yf
    raw = yf.download(list(YAHOO.values()), period="max", interval="1d", group_by="ticker", auto_adjust=False, progress=False, threads=True)
    out = {}
    for k, t in YAHOO.items():
        if t in raw.columns.get_level_values(0):
            s = raw[t]["Close"].dropna()
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            out[k] = s[~s.index.duplicated(keep="last")]
    return out


def fetch_fred():
    out = {}
    for k, sid in FRED.items():
        try:
            df = pd.read_csv(io.BytesIO(get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}").content))
            df.columns = ["date", "v"]
            s = pd.to_numeric(df["v"], errors="coerce")
            s.index = pd.to_datetime(df["date"])
            out[k] = s.dropna()
        except Exception as e:
            print(f"FRED {sid}: {e}")
    return out


def fetch_cot():
    out = {}
    for k, code in COT_CODES.items():
        try:
            rows = get("https://publicreporting.cftc.gov/resource/6dca-aqww.json",
                       params={"cftc_contract_market_code": code, "$order": "report_date_as_yyyy_mm_dd DESC", "$limit": 1200}).json()
            df = pd.DataFrame(rows)
            net = (pd.to_numeric(df["noncomm_positions_long_all"]) - pd.to_numeric(df["noncomm_positions_short_all"])) / pd.to_numeric(df["open_interest_all"])
            net.index = pd.to_datetime(df["report_date_as_yyyy_mm_dd"]).dt.normalize()
            # published on Friday for Tuesday's data: shift 3 days so history only uses what was known
            out[k] = (net.sort_index() * 100).shift(3, freq="D")
        except Exception as e:
            print(f"CFTC {code}: {e}")
    return out


def fetch_gld():
    try:
        txt = get("https://www.spdrgoldshares.com/assets/dynamic/GLD/GLD_US_archive_EN.csv").text
        df = pd.read_csv(io.StringIO(txt), skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]
        col = next(c for c in df.columns if "Tonnes" in c)
        s = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
        s.index = pd.to_datetime(df[df.columns[0]], errors="coerce")
        return s[s.index.notna()].dropna().sort_index()
    except Exception as e:
        print(f"GLD holdings: {e}")
        return None


def fetch_news():
    items, seen = [], set()
    for q in NEWS_Q:
        try:
            xml = get("https://news.google.com/rss/search", params={"q": q + " when:7d", "hl": "en-IN", "gl": "IN", "ceid": "IN:en"}).content
            for it in ET.fromstring(xml).iter("item"):
                title = (it.findtext("title") or "").strip()
                key = re.sub(r"\W+", "", title.lower())[:60]
                if not title or key in seen:
                    continue
                seen.add(key)
                try:
                    d = parsedate_to_datetime(it.findtext("pubDate")).strftime("%Y-%m-%d")
                except Exception:
                    d = ""
                low = " " + title.lower() + " "
                tags = [t for t, words in TAGS.items() if any(w in low for w in words)]
                items.append({"title": title, "link": it.findtext("link"), "date": d, "tags": tags[:2],
                              "metal": "silver" if "silver" in low and "gold" not in low else "gold" if "gold" in low else "both"})
        except Exception as e:
            print(f"news '{q}': {e}")
    return sorted(items, key=lambda x: x["date"], reverse=True)[:25]


# ------------------------------------------------------------------ factors
def sgn(x, thr):
    return x.where(x.isna(), np.where(x > thr, 1.0, np.where(x < -thr, -1.0, 0.0)))


def build_frame(y, fred, cot, gld):
    idx = y["gold"].index
    df = pd.DataFrame(index=idx)
    if "cpi" in fred:  # CPI for a month is published ~2 weeks later: only use it once it was known
        fred = {**fred, "cpi": fred["cpi"].shift(45, freq="D")}
    for k, s in {**y, **fred, **cot}.items():
        df[k] = s.reindex(idx.union(s.index)).ffill().reindex(idx)
    if gld is not None:
        df["gld_t"] = gld.reindex(idx.union(gld.index)).ffill().reindex(idx)
    return df


def factors(df, metal):
    """Each factor: +1 helping, -1 hurting, 0 neutral, NaN unavailable. Uses only data up to each day."""
    p = df[metal]
    ma50, ma200 = p.rolling(50).mean(), p.rolling(200).mean()
    F = {}
    has = lambda c: c in df and df[c].notna().sum() > 300
    if has("ry"):
        F["ry3"] = sgn(-(df.ry - df.ry.shift(63)), 0.15); F["ry6"] = sgn(-(df.ry - df.ry.shift(126)), 0.25)
    if has("dxy"):
        F["dxy3"] = sgn(-(df.dxy / df.dxy.shift(63) - 1), 0.02); F["dxy6"] = sgn(-(df.dxy / df.dxy.shift(126) - 1), 0.03)
    if has("ff"):
        F["fed"] = sgn(-(df.ff - df.ff.shift(126)), 0.25); F["fed12"] = sgn(-(df.ff - df.ff.shift(252)), 0.5)
    F["trend"] = pd.Series(np.where((p > ma50) & (p > ma200), 1.0, np.where((p < ma50) & (p < ma200), -1.0, 0.0)), index=p.index).where(ma200.notna())
    F["mom3"] = sgn(p / p.shift(63) - 1, 0.02)
    F["mom12"] = sgn(p.shift(21) / p.shift(252) - 1, 0.03)
    F["mom24"] = sgn(p.shift(21) / p.shift(504) - 1, 0.05)
    cot = "cot_g" if metal == "gold" else "cot_s"
    if has(cot):
        pct = df[cot].rolling(756, min_periods=250).rank(pct=True)
        F["pos"] = pd.Series(np.where(pct > 0.85, -1.0, np.where(pct < 0.15, 1.0, 0.0)), index=p.index).where(pct.notna())
    if metal == "gold" and has("gld_t"):
        F["etf"] = sgn(df.gld_t / df.gld_t.shift(21) - 1, 0.01)
    if has("usdinr"):
        F["inr"] = sgn(df.usdinr / df.usdinr.shift(63) - 1, 0.01)
    if has("vix"):
        F["fear"] = pd.Series(np.where(df.vix > 25, 1.0, 0.0), index=p.index).where(df.vix.notna())
    if has("cpi"):
        real = p / df.cpi
        pct = real.expanding(min_periods=2500).rank(pct=True)
        F["value"] = pd.Series(np.where(pct > 0.85, -1.0, np.where(pct < 0.30, 1.0, 0.0)), index=p.index).where(pct.notna())
    if metal == "silver":
        ratio = df.gold / df.silver
        F["ratio"] = pd.Series(np.where(ratio > 85, 1.0, np.where(ratio < 65, -1.0, 0.0)), index=p.index).where(ratio.notna())
        g = df.gold
        F["gold_trend"] = pd.Series(np.where((g > g.rolling(50).mean()) & (g > g.rolling(200).mean()), 1.0,
                                             np.where((g < g.rolling(50).mean()) & (g < g.rolling(200).mean()), -1.0, 0.0)), index=p.index)
        if has("copper"):
            F["copper"] = sgn(df.copper / df.copper.shift(63) - 1, 0.05)
    return pd.DataFrame(F)


WEIGHTS = {
    "gold": {"1–3 months": {"trend": .25, "mom3": .15, "dxy3": .15, "ry3": .15, "pos": .15, "fed": .10, "inr": .05},
             "6–12 months": {"mom12": .25, "ry6": .25, "dxy6": .15, "fed": .15, "etf": .10, "cb": .10},
             "1–2 years": {"cb": .35, "fed12": .35, "mom24": .30},
             "3–5 years": {"value": .6, "cb": .4}},
    "silver": {"1–3 months": {"trend": .25, "gold_trend": .15, "mom3": .10, "dxy3": .10, "ry3": .10, "pos": .10, "copper": .15, "inr": .05},
               "6–12 months": {"mom12": .25, "ry6": .20, "dxy6": .15, "copper": .20, "fed": .10, "ratio": .10},
               "1–2 years": {"ratio": .35, "fed12": .35, "mom24": .30},
               "3–5 years": {"value": .5, "ratio": .5}}}
CONF_CAP = {"1–3 months": "Medium", "6–12 months": "Medium", "1–2 years": "Low", "3–5 years": "Context only"}


def composite(F, weights, cb=None):
    num = pd.Series(0.0, index=F.index); den = pd.Series(0.0, index=F.index)
    for k, w in weights.items():
        if k == "cb":
            if cb is None:
                continue
            s = pd.Series(np.nan, index=F.index); s.iloc[-1] = cb
        elif k in F:
            s = F[k]
        else:
            continue
        ok = s.notna()
        num[ok] += w * s[ok]; den[ok] += w
    return (num / den).where(den >= 0.5)


def label(x, long=False):
    if x is None or np.isnan(x):
        return "No data"
    if long:
        return "Favourable" if x > 0.25 else "Unfavourable" if x < -0.25 else "Neutral"
    return "Positive" if x > 0.25 else "Negative" if x < -0.25 else "Neutral"


def track_record(price, comp, days):
    """Month-end readings vs the metal's return over the following `days` trading days."""
    fwd = price.shift(-days) / price - 1
    m = pd.DataFrame({"c": comp, "f": fwd}).dropna()
    m = m.groupby(m.index.to_period("M")).tail(1)
    if len(m) < 24:
        return None
    m["lab"] = m["c"].map(label)
    out = {"from": m.index[0].strftime("%Y"), "to": m.index[-1].strftime("%Y"), "months": len(m),
           "all": {"avg_pct": round(m.f.mean() * 100, 1), "up_pct": round((m.f > 0).mean() * 100)}}
    for lab in ("Positive", "Neutral", "Negative"):
        g = m[m.lab == lab].f
        out[lab] = {"n": len(g), "avg_pct": round(g.mean() * 100, 1) if len(g) else None, "up_pct": round((g > 0).mean() * 100) if len(g) else None}
    return out


# ------------------------------------------------------------------ driver descriptions
def describe(df, metal, F):
    last = df.iloc[-1]; d = []
    ch = lambda c, n: (df[c].iloc[-1] / df[c].iloc[-1 - n] - 1) * 100
    st = lambda k: {1.0: "help", -1.0: "hurt", 0.0: "neutral"}.get(F[k].iloc[-1], "neutral") if k in F and not np.isnan(F[k].iloc[-1]) else None
    add = lambda name, key, val, why: d.append({"name": name, "effect": st(key), "value": val, "why": why}) if st(key) else None
    if metal == "silver":
        add("Gold trend", "gold_trend", "Gold above its 50- and 200-day averages" if F["gold_trend"].iloc[-1] > 0 else "Gold below its averages" if F["gold_trend"].iloc[-1] < 0 else "Gold mixed",
            "Silver usually follows gold, with bigger swings.")
        add("Gold/silver ratio", "ratio", f"{last.gold / last.silver:.1f} ({'silver cheap vs gold' if last.gold / last.silver > 85 else 'silver expensive vs gold' if last.gold / last.silver < 65 else 'normal range'})",
            "Above 85 has often been followed by silver outperforming gold; below 65 the opposite.")
        if "copper" in F:
            add("Industrial demand (copper)", "copper", f"Copper {ch('copper', 63):+.1f}% in 3 months", "About half of silver demand is industrial; copper is a live proxy for it.")
    if "ry3" in F:
        add("US real interest rates", "ry3", f"{last.ry:.2f}%, {df.ry.iloc[-1] - df.ry.iloc[-64]:+.2f} pts in 3 months",
            "Rising real rates make metals (which pay no interest) less attractive. Gold's biggest single driver.")
    if "dxy3" in F:
        add("US dollar", "dxy3", f"Dollar index {last.dxy:.1f}, {ch('dxy', 63):+.1f}% in 3 months", "A stronger dollar usually pushes gold and silver down.")
    if "fed" in F:
        add("Fed policy", "fed", f"Fed funds {last.ff:.2f}%, {df.ff.iloc[-1] - df.ff.iloc[-127]:+.2f} pts in 6 months", "Rate hikes hurt metals; cuts help.")
    p = df[metal]
    add("Price trend", "trend", f"{'Above' if p.iloc[-1] > p.rolling(50).mean().iloc[-1] else 'Below'} 50-day, {'above' if p.iloc[-1] > p.rolling(200).mean().iloc[-1] else 'below'} 200-day average · {ch(metal, 63):+.1f}% in 3 months",
        "Metal trends tend to persist for months.")
    add("12-month momentum", "mom12", f"{(p.iloc[-22] / p.iloc[-253] - 1) * 100:+.1f}% over the past year", "Rising over the past year has tended to continue into the next.")
    cot = "cot_g" if metal == "gold" else "cot_s"
    if "pos" in F:
        add("Speculative positioning", "pos", f"Speculators net long {last[cot]:.0f}% of open interest",
            "Very crowded long positions often come before sharp falls; very low positioning before rebounds.")
    if "etf" in F:
        add("ETF demand", "etf", f"Largest gold ETF holdings {ch('gld_t', 21):+.1f}% in a month ({last.gld_t:,.0f} t)", "Money flowing into gold funds supports prices.")
    if "inr" in F:
        add("USD/INR", "inr", f"₹{last.usdinr:.2f} per $, {ch('usdinr', 63):+.1f}% in 3 months", "A weaker rupee raises metal prices in India.")
    if "fear" in F:
        add("Fear and stress", "fear", f"VIX {last.vix:.1f}", "Gold gets safe-haven buying when markets are stressed (VIX above 25).")
    return d


# ------------------------------------------------------------------ main entry
def run(manual=None, holdings=None, portfolio_value=None):
    manual = manual or {}
    y = fetch_yahoo()
    if "gold" not in y or "silver" not in y:
        return {"error": "Couldn't download gold and silver prices today."}
    fred, cot, gld = fetch_fred(), fetch_cot(), fetch_gld()
    df = build_frame(y, fred, cot, gld)
    cb = manual.get("cb_signal")  # +1 / 0 / -1 from the quarterly central bank update
    news = fetch_news()
    out = {"date": df.index[-1].strftime("%Y-%m-%d"), "news": news, "manual": manual,
           "sources": {"fred": sorted(fred), "cot": sorted(cot), "gld": gld is not None, "news": len(news)}}
    for metal in ("gold", "silver"):
        F = factors(df, metal)
        p = df[metal]; last = df.iloc[-1]
        inr = last.usdinr if "usdinr" in df and not np.isnan(last.usdinr) else None
        hz, tr = [], {}
        for h, w in WEIGHTS[metal].items():
            c = composite(F, w, cb if metal == "gold" or "cb" in w else None)
            v = c.iloc[-1]
            long = h == "3–5 years"
            agree = abs(v) if not np.isnan(v) else 0
            conf = CONF_CAP[h] if h in ("3–5 years",) else ("Low" if agree < 0.4 or CONF_CAP[h] == "Low" else CONF_CAP[h])
            used = [k for k in w if (k == "cb" and cb is not None) or (k in F and not np.isnan(F[k].iloc[-1]))]
            hz.append({"horizon": h, "score": None if np.isnan(v) else round(float(v), 2), "label": label(v, long), "confidence": conf, "drivers": used})
            if h in ("1–3 months", "6–12 months"):
                tr[h] = track_record(p, composite(F, w), 63 if h == "1–3 months" else 252)
        series = p.iloc[-260:]
        out[metal] = {"price_usd": round(float(p.iloc[-1]), 2), "day_pct": round(float((p.iloc[-1] / p.iloc[-2] - 1) * 100), 2),
                      "chg_1m": round(float((p.iloc[-1] / p.iloc[-22] - 1) * 100), 1), "chg_1y": round(float((p.iloc[-1] / p.iloc[-253] - 1) * 100), 1),
                      "price_inr": round(float(p.iloc[-1] * inr / OZ * (10 if metal == "gold" else 1000))) if inr else None,
                      "inr_unit": "10 g" if metal == "gold" else "1 kg", "high_1y": round(float(series.max()), 2), "low_1y": round(float(series.min()), 2),
                      "spark": [round(float(x), 2) for x in series.iloc[::5]], "horizons": hz, "drivers": describe(df, metal, F), "track": tr}
    if cb is not None:
        out["gold"]["drivers"].append({"name": "Central bank buying", "effect": {1: "help", -1: "hurt"}.get(cb, "neutral"),
            "value": f"{manual.get('cb_tonnes', '?')} t in {manual.get('cb_quarter', 'latest quarter')}, {manual.get('cb_yoy', 0):+g}% vs a year ago (updated {manual.get('cb_date', '?')})",
            "why": "Steady official buying supports prices, especially on dips. You update this each quarter from the World Gold Council."})
    out["ratio"] = round(float(df.gold.iloc[-1] / df.silver.iloc[-1]), 1)
    out["advice"] = advice(out, holdings or [], portfolio_value)
    return out


def advice(out, holdings, pv):
    """Plain-language view for the user's gold/silver holdings. Uses a 10-15% allocation guideline."""
    mh = [h for h in holdings if h.get("group") == "Gold / Silver" or any(k in h["symbol"] for k in ("GOLD", "SILVER"))]
    val = sum(h["value"] for h in mh)
    w = val / pv * 100 if pv else None
    res = {"weight_pct": round(w, 1) if w is not None else None, "value": round(val), "holdings": [
        {"symbol": h["symbol"], "broker": h.get("broker", "Kite"), "qty": h["qty"], "pnl_pct": round(h["pnl_pct"], 1),
         "metal": "silver" if "SILVER" in h["symbol"] else "gold"} for h in mh]}
    for metal in ("gold", "silver"):
        s = {x["horizon"]: x["label"] for x in out[metal]["horizons"]}
        short, mid = s.get("1–3 months"), s.get("6–12 months")
        over, under = w is not None and w > 15, w is not None and w < 10
        if over and short == "Negative":
            t = "Overweight and short-term outlook negative: don't add. Consider trimming on rallies to bring metals back toward 10–15%."
        elif over:
            t = "Overweight: hold, but don't add until metals are back under 15% of your portfolio."
        elif under and mid == "Positive" and short != "Negative":
            t = "Underweight and the 6–12 month outlook is positive: adding gradually (small monthly amounts) fits the guideline."
        elif short == "Negative" and mid == "Negative":
            t = "Short- and medium-term outlooks negative: wait before adding."
        else:
            t = "Mixed outlook: hold. Add only gradually if you're below 10–15% in metals."
        res[metal] = t
    return res
