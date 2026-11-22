"""Daily dashboard data builder.

Reads holdings (encrypted holdings.enc.json, or the HOLDINGS_JSON secret on first run),
fetches prices (NSE bhavcopy -> BSE bhavcopy -> Yahoo), computes indicators, signals,
portfolio health and market data, keeps a daily history, and writes everything
encrypted with DASHBOARD_PASSWORD.
"""
import base64
import io
import json
import os
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IST = timezone(timedelta(hours=5, minutes=30))
PBKDF2_ITERS = 310000

MARKET = [  # (name, yahoo, kind)
    ("Nifty 50", "^NSEI", "index"), ("Sensex", "^BSESN", "index"), ("Bank Nifty", "^NSEBANK", "index"),
    ("India VIX", "^INDIAVIX", "index"), ("USD/INR", "INR=X", "fx"),
    ("Gold $/oz", "GC=F", "metal"), ("Silver $/oz", "SI=F", "metal"),
    ("Nifty IT", "^CNXIT", "sector"), ("Nifty Pharma", "^CNXPHARMA", "sector"),
    ("Nifty Auto", "^CNXAUTO", "sector"), ("Nifty FMCG", "^CNXFMCG", "sector"),
    ("Nifty Metal", "^CNXMETAL", "sector"), ("Nifty Energy", "^CNXENERGY", "sector"),
    ("Nifty PSU Bank", "^CNXPSUBANK", "sector"), ("Nifty Realty", "^CNXREALTY", "sector"),
    ("Nifty Infra", "^CNXINFRA", "sector"), ("Nifty Fin Service", "NIFTY_FIN_SERVICE.NS", "sector"),
]
METAL_LEVELS = {"GC=F": [("Support", 3690), ("Bullish above", 3948)], "SI=F": [("Key level", 40)]}


# ---------------------------------------------------------------- crypto
def _key(password, salt, iters):
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    return PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=iters).derive(password.encode())


def encrypt(text, password):
    """AES-256-GCM, PBKDF2-SHA256 key. Matches the WebCrypto code in index.html."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    salt, iv = os.urandom(16), os.urandom(12)
    ct = AESGCM(_key(password, salt, PBKDF2_ITERS)).encrypt(iv, text.encode(), None)
    b64 = lambda b: base64.b64encode(b).decode()
    return {"v": 1, "iter": PBKDF2_ITERS, "salt": b64(salt), "iv": b64(iv), "ct": b64(ct)}


def decrypt(blob, password):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    d = lambda s: base64.b64decode(s)
    key = _key(password, d(blob["salt"]), blob["iter"])
    return AESGCM(key).decrypt(d(blob["iv"]), d(blob["ct"]), None).decode()


def read_enc(name, password, default=None):
    p = ROOT / name
    if not p.exists():
        return default
    return json.loads(decrypt(json.loads(p.read_text()), password))


def write_enc(name, obj, password):
    (ROOT / name).write_text(json.dumps(encrypt(json.dumps(obj, default=float, separators=(",", ":")), password)))


# ---------------------------------------------------------------- indicators
def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def atr(df, n):
    pc = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def adx(df, n=14):
    h, l = df["High"], df["Low"]
    upm, dnm = h.diff(), -l.diff()
    pdm = pd.Series(np.where((upm > dnm) & (upm > 0), upm, 0.0), index=df.index)
    ndm = pd.Series(np.where((dnm > upm) & (dnm > 0), dnm, 0.0), index=df.index)
    a = atr(df, n)
    pdi = 100 * pdm.ewm(alpha=1 / n, adjust=False).mean() / a
    ndi = 100 * ndm.ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def supertrend(df, n=10, mult=3.0):
    a = atr(df, n)
    hl2 = (df["High"] + df["Low"]) / 2
    ub, lb = (hl2 + mult * a).values, (hl2 - mult * a).values
    c = df["Close"].values
    fu, fl = ub.copy(), lb.copy()
    direction = np.ones(len(c))
    for i in range(1, len(c)):
        fu[i] = ub[i] if (ub[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lb[i] if (lb[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        if direction[i - 1] == 1:
            direction[i] = -1 if c[i] < fl[i] else 1
        else:
            direction[i] = 1 if c[i] > fu[i] else -1
    return int(direction[-1]), float(np.where(direction == 1, fl, fu)[-1])


def clean(df):
    if df is None:
        return None
    df = df.dropna(subset=["Close"]).copy()
    if df.empty:
        return None
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="last")]
    if "Volume" in df:
        flat = (df["Volume"].fillna(0) == 0) & (df["High"] == df["Low"])
        if (~flat).sum() > 1:
            df = df[~flat]
    return df


def r2(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 2)


def indicators(df, chart_days=100):
    df = clean(df)
    if df is None:
        raise ValueError("no price data from NSE, BSE or Yahoo")
    c = df["Close"]
    out = {"close": float(c.iloc[-1]), "prev_close": float(c.iloc[-2]) if len(c) > 1 else None,
           "date": df.index[-1].strftime("%Y-%m-%d"),
           "high52": r2(df["High"].tail(250).max()), "low52": r2(df["Low"].tail(250).min()),
           "ema50": None, "sma200": None, "rsi": None, "adx": None, "st_dir": 0, "st_level": None, "atr": None}
    out["day_pct"] = (out["close"] / out["prev_close"] - 1) * 100 if out["prev_close"] else 0.0
    ema = c.ewm(span=50, adjust=False).mean()
    sma = c.rolling(200).mean()
    if len(c) >= 15:
        d, lvl = supertrend(df)
        out["atr"] = r2(atr(df, 14).iloc[-1])
        out.update({"ema50": r2(ema.iloc[-1]), "sma200": r2(sma.iloc[-1]) if len(c) >= 200 else None,
                    "rsi": r2(rsi(c).iloc[-1]), "adx": r2(adx(df).iloc[-1]), "st_dir": d, "st_level": r2(lvl)})
    t = df.tail(chart_days)
    out["chart"] = {"d": [x.strftime("%Y-%m-%d") for x in t.index],
                    "o": [r2(x) for x in t["Open"]], "h": [r2(x) for x in t["High"]],
                    "l": [r2(x) for x in t["Low"]], "c": [r2(x) for x in t["Close"]],
                    "ema": [r2(x) for x in ema.tail(chart_days)],
                    "sma": [r2(x) for x in sma.tail(chart_days)]}
    return out


# ---------------------------------------------------------------- signals
SIGNALS = {"add": "Add zone", "hold": "Hold · uptrend", "wait": "Wait", "noadd": "Don't add", "exit": "Exit alert"}


def auto_levels(h, ind):
    """Rules used for holdings without hand-set levels; recalculated daily."""
    c, ema, st, d = ind["close"], ind["ema50"], ind["st_level"], ind["st_dir"]
    if d == 0 or ema is None:
        return {**h, "auto": True, "note": "Not enough price history for levels yet."}
    h = {**h, "auto": True}
    if d == -1:
        h.update(add_above=st, stop_below=None, note=f"Downtrend. Don't add until a close above {st:,.2f}.")
    elif c < ema:
        h.update(add_above=ema, stop_below=st,
                 note=f"Recovering. Add only after a close above the 50-EMA ({ema:,.2f}); exit below {st:,.2f}.")
    else:
        h.update(add_above=None, stop_below=st, note=f"Uptrend confirmed. Hold; trail the stop at {st:,.2f}.")
    return h


def signal(h, ind):
    c = ind["close"]
    if h.get("stop_below") and c < h["stop_below"]:
        return "exit"
    if h.get("add_above") and c > h["add_above"]:
        return "add"
    if ind["st_dir"] == -1:
        return "noadd"
    if ind["st_dir"] == 1 and not h.get("add_above"):
        return "hold"
    return "wait"


# ---------------------------------------------------------------- data sources
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
           "(KHTML, like Gecko) Chrome/124.0 Safari/537.36", "Accept": "*/*",
           "Accept-Language": "en-US,en;q=0.9"}
BHAV_URLS = {
    "NSE": ("https://www.nseindia.com/",
            "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{d}_F_0000.csv.zip"),
    "BSE": ("https://www.bseindia.com/",
            "https://www.bseindia.com/download/BhavCopy/Equity/BhavCopy_BSE_CM_0_0_0_{d}_F_0000.CSV"),
}
SERIES_PREF = ["EQ", "BE", "BZ", "SM", "ST", "A", "B", "T", "X", "M", "MT", "XT", "Z"]


def fetch_bhavcopy(exchange, max_back=7):
    """Official end-of-day file (UDiFF). Returns (date, {symbol: row}, {isin: symbol}, raw df)."""
    import requests
    home, url = BHAV_URLS[exchange]
    sess = requests.Session()
    sess.headers.update({**HEADERS, "Referer": home})
    try:
        sess.get(home, timeout=15)
    except Exception:
        pass
    today = datetime.now(IST).date()
    for back in range(max_back + 1):
        day = today - timedelta(days=back)
        if day.weekday() >= 5:
            continue
        try:
            r = sess.get(url.format(d=day.strftime("%Y%m%d")), timeout=30)
        except Exception as e:
            print(f"{exchange} {day}: {e}")
            continue
        if r.status_code != 200 or len(r.content) < 100:
            print(f"{exchange} {day}: HTTP {r.status_code}")
            continue
        body = r.content
        if body[:2] == b"PK":
            with zipfile.ZipFile(io.BytesIO(body)) as z:
                body = z.read(z.namelist()[0])
        df = pd.read_csv(io.BytesIO(body))
        df.columns = [c.strip() for c in df.columns]
        df["TckrSymb"] = df["TckrSymb"].astype(str).str.strip()
        df["SctySrs"] = df["SctySrs"].astype(str).str.strip()
        df["_rank"] = df["SctySrs"].map(lambda s: SERIES_PREF.index(s) if s in SERIES_PREF else 99)
        best = df.sort_values("_rank").drop_duplicates("TckrSymb")
        rows = {r_.TckrSymb: {"open": float(r_.OpnPric), "high": float(r_.HghPric), "low": float(r_.LwPric),
                              "close": float(r_.ClsPric), "prev": float(r_.PrvsClsgPric),
                              "vol": float(getattr(r_, "TtlTradgVol", 0) or 0)}
                for r_ in best.itertuples()}
        isin = {str(r_.ISIN).strip(): r_.TckrSymb for r_ in best.itertuples() if isinstance(r_.ISIN, str)}
        print(f"{exchange} bhavcopy {day}: {len(rows)} symbols")
        return pd.Timestamp(day), rows, isin, df
    return None, {}, {}, None


def merge_official(hist, date, row):
    new = pd.DataFrame({"Open": [row["open"]], "High": [row["high"]], "Low": [row["low"]],
                        "Close": [row["close"]], "Volume": [1.0]}, index=[date])
    hist = clean(hist)
    if hist is None:
        prev = pd.DataFrame({"Open": [row["prev"]], "High": [row["prev"]], "Low": [row["prev"]],
                             "Close": [row["prev"]], "Volume": [1.0]}, index=[date - pd.Timedelta(days=1)])
        return pd.concat([prev, new])
    hist = hist[hist.index < date]
    if len(hist):
        hist.iloc[-1, hist.columns.get_loc("Close")] = row["prev"]
    return pd.concat([hist[["Open", "High", "Low", "Close", "Volume"]], new])


def fetch(tickers):
    import yfinance as yf
    raw = yf.download(tickers, period="2y", interval="1d", group_by="ticker",
                      auto_adjust=False, progress=False, threads=True)
    return {t: raw[t] for t in tickers if t in raw.columns.get_level_values(0)}


def fetch_fundamentals(ticker):
    import yfinance as yf
    i = yf.Ticker(ticker).info or {}
    dy = i.get("dividendYield")
    return {"pe": r2(i.get("trailingPE")), "pb": r2(i.get("priceToBook")),
            "roe": r2(i["returnOnEquity"] * 100) if i.get("returnOnEquity") is not None else None,
            "de": r2(i["debtToEquity"] / 100) if i.get("debtToEquity") is not None else None,
            "div": r2(dy if dy and dy > 1 else (dy or 0) * 100) if dy is not None else None,
            "mcap_cr": r2(i["marketCap"] / 1e7) if i.get("marketCap") else None,
            "sector": i.get("sector"), "industry": i.get("industry"), "name": i.get("longName") or i.get("shortName")}


def cap_bucket(mcap_cr):
    if not mcap_cr:
        return None
    return "Large" if mcap_cr >= 100000 else "Mid" if mcap_cr >= 33000 else "Small"


# ---------------------------------------------------------------- main
def load_holdings(password):
    cfg = read_enc("holdings.enc.json", password)
    if cfg is None:
        raw = os.environ.get("HOLDINGS_JSON")
        if not raw and (ROOT / "holdings.json").exists():
            raw = (ROOT / "holdings.json").read_text()
        if not raw:
            sys.exit("No holdings found (holdings.enc.json or HOLDINGS_JSON secret).")
        cfg = json.loads(raw)
        write_enc("holdings.enc.json", cfg, password)  # first run only; afterwards the page owns this file
        print("Created holdings.enc.json from the HOLDINGS_JSON secret.")
    return cfg


def main():
    password = os.environ.get("DASHBOARD_PASSWORD")
    if not password:
        sys.exit("DASHBOARD_PASSWORD is not set - refusing to publish unencrypted data.")
    cfg = load_holdings(password)
    state = read_enc("state.enc.json", password, {}) or {}
    state.setdefault("history", []); state.setdefault("signals", {})
    state.setdefault("events", []); state.setdefault("fund", {})

    bhav = {ex: fetch_bhavcopy(ex) for ex in ("NSE", "BSE")}
    isin_map = {**bhav["BSE"][2], **bhav["NSE"][2]}
    if bhav["NSE"][2]:
        (ROOT / "isin.json").write_text(json.dumps(bhav["NSE"][2], separators=(",", ":")))

    holdings = []
    for h in cfg["holdings"]:
        h = dict(h)
        if not h.get("symbol") and h.get("isin") in isin_map:
            h["symbol"] = isin_map[h["isin"]]
        if not h.get("symbol"):
            print(f"Could not resolve {h}")
            continue
        h.setdefault("yahoo", h["symbol"] + ".NS")
        h.setdefault("group", "Gold / Silver" if any(k in h["symbol"] for k in ("GOLD", "SILVER")) else "Stock")
        holdings.append(h)

    # one plan per stock: hand-set levels on any account apply to the same stock on every account
    custom = {h["symbol"]: h for h in holdings if h.get("add_above") is not None or h.get("stop_below") is not None}
    for h in holdings:
        c = custom.get(h["symbol"])
        if c is not None and c is not h and h.get("add_above") is None and h.get("stop_below") is None:
            h.update(add_above=c.get("add_above"), stop_below=c.get("stop_below"), note=c.get("note", ""), group=c.get("group", h["group"]))

    tickers = sorted({h["yahoo"] for h in holdings} | {m[1] for m in MARKET})
    try:
        data = fetch(tickers)
    except Exception as e:
        print(f"Yahoo failed: {e}")
        data = {}

    today = datetime.now(IST).strftime("%Y-%m-%d")
    rows, errors, charts = [], [], {}
    for h in holdings:
        hist, source = data.get(h["yahoo"]), "Yahoo"
        for ex in ("NSE", "BSE"):
            bdate, brows = bhav[ex][0], bhav[ex][1]
            key = h["symbol"] if ex == "NSE" else h.get("bse", h["symbol"])
            if bdate is not None and key in brows:
                hist, source = merge_official(hist, bdate, brows[key]), ex
                break
        try:
            ind = indicators(hist)
        except Exception as e:
            errors.append(f"{h['symbol']}: {e}")
            continue
        charts[h["symbol"]] = ind.pop("chart")
        if h.get("add_above") is None and h.get("stop_below") is None and h["group"] != "Gold / Silver":
            h = auto_levels(h, ind)
        sig = signal(h, ind)
        f = state["fund"].get(h["symbol"])
        if h["group"] != "Gold / Silver" and (not f or f.get("t", "") < (datetime.now(IST) - timedelta(days=7)).strftime("%Y-%m-%d")):
            try:
                f = {**fetch_fundamentals(h["yahoo"]), "t": today}
                state["fund"][h["symbol"]] = f
            except Exception as e:
                print(f"fundamentals {h['symbol']}: {e}")
        f = f or {}
        invested, value = h["qty"] * h["avg"], h["qty"] * ind["close"]
        sector = "Gold and silver" if h["group"] == "Gold / Silver" else (f.get("sector") or "Other")
        rows.append({**h, **ind, "source": source, "invested": invested, "value": value,
                     "pnl": value - invested, "pnl_pct": (value / invested - 1) * 100,
                     "day_pnl": h["qty"] * (ind["close"] - ind["prev_close"]) if ind["prev_close"] else 0.0,
                     "breakeven_pct": (h["avg"] / ind["close"] - 1) * 100,
                     "sig": sig, "signal": SIGNALS[sig], "sector": sector,
                     "cap": None if h["group"] == "Gold / Silver" else cap_bucket(f.get("mcap_cr")),
                     "fund": {k: v for k, v in f.items() if k != "t"}})
        print(f"{h['symbol']:<11} {ind['date']} close={ind['close']:.2f} prev={ind['prev_close']} [{source}] {sig}")

        key = f"{h.get('broker', 'Kite')}:{h['symbol']}"
        fs = state.setdefault("first_seen", {})
        fs.setdefault(key, {"date": ind["date"]})  # first day this holding appeared (kept while you hold it)
        rows[-1]["first_seen"] = fs[key]["date"]
        old = state["signals"].get(key)
        if old and old["sig"] != sig:
            state["events"].insert(0, {"date": ind["date"], "symbol": h["symbol"], "broker": h.get("broker", "Kite"),
                                       "from": old["sig"], "to": sig, "price": ind["close"]})
        state["signals"][key] = {"sig": sig, "since": old["since"] if old and old["sig"] == sig else ind["date"]}
    state["events"] = state["events"][:60]
    held = {f"{r.get('broker', 'Kite')}:{r['symbol']}" for r in rows}
    state["first_seen"] = {k: v for k, v in state.get("first_seen", {}).items() if k in held}

    # market
    market = []
    for name, yt, kind in MARKET:
        try:
            ind = indicators(data.get(yt), chart_days=60)
            market.append({"name": name, "kind": kind, "close": ind["close"], "day_pct": ind["day_pct"],
                           "st_dir": ind["st_dir"], "spark": ind["chart"]["c"],
                           "levels": [{"label": l, "value": v} for l, v in METAL_LEVELS.get(yt, [])]})
        except Exception as e:
            errors.append(f"{name}: {e}")
    breadth, movers = None, None
    raw = bhav["NSE"][3]
    if raw is not None:
        eq = raw[raw["SctySrs"] == "EQ"].copy()
        eq["chg"] = (eq["ClsPric"] / eq["PrvsClsgPric"] - 1) * 100
        breadth = {"up": int((eq["chg"] > 0).sum()), "down": int((eq["chg"] < 0).sum()),
                   "flat": int((eq["chg"] == 0).sum())}
        liq = eq[(eq["ClsPric"] >= 20) & (eq.get("TtlTrfVal", 0) >= 1e8)]
        pick = lambda d: [{"symbol": r_.TckrSymb, "close": float(r_.ClsPric), "chg": float(r_.chg)} for r_ in d.itertuples()]
        movers = {"gainers": pick(liq.nlargest(5, "chg")), "losers": pick(liq.nsmallest(5, "chg"))}

    # summary, brokers, history
    def agg(rs):
        inv, val = sum(r["invested"] for r in rs), sum(r["value"] for r in rs)
        day = sum(r["day_pnl"] for r in rs)
        prev = val - day
        return {"invested": inv, "value": val, "pnl": val - inv, "pnl_pct": (val / inv - 1) * 100 if inv else 0,
                "day": day, "day_pct": day / prev * 100 if prev else 0}
    summary = agg(rows)
    brokers = {b: agg([r for r in rows if r.get("broker", "Kite") == b])
               for b in sorted({r.get("broker", "Kite") for r in rows})}
    price_date = max((r["date"] for r in rows), default=today)
    hist = [x for x in state["history"] if x["d"] != price_date]
    hist.append({"d": price_date, "v": round(summary["value"]), "i": round(summary["invested"]),
                 **{b: round(x["value"]) for b, x in brokers.items()}})
    state["history"] = sorted(hist, key=lambda x: x["d"])[-1500:]

    try:
        import scanner
        scan = scanner.run(ROOT, state, bhav["NSE"] if bhav["NSE"][0] is not None else None)
    except Exception as e:
        import traceback; traceback.print_exc()
        scan = {"error": f"Scanner failed today: {e}"}

    try:
        import metals
        met = metals.run(manual=cfg.get("metals"), holdings=rows, portfolio_value=summary["value"])
    except Exception as e:
        import traceback; traceback.print_exc()
        met = {"error": f"Gold and silver data failed today: {e}"}

    try:
        import etfs
        etf = etfs.run(state, bhav["NSE"] if bhav["NSE"][0] is not None else None, met, rows)
    except Exception as e:
        import traceback; traceback.print_exc()
        etf = {"error": f"ETF data failed today: {e}"}

    out = {"updated": datetime.now(IST).strftime("%d %b %Y, %I:%M %p IST"), "price_date": price_date, "scanner": scan, "metals": met, "etfs": etf,
           "sources": {ex: (bhav[ex][0].strftime("%Y-%m-%d") if bhav[ex][0] is not None else None) for ex in bhav},
           "summary": summary, "brokers": brokers, "holdings": rows, "charts": charts,
           "market": market, "breadth": breadth, "movers": movers,
           "history": state["history"], "events": state["events"],
           "since": {k: v["since"] for k, v in state["signals"].items()}, "errors": errors}
    write_enc("data.enc.json", out, password)
    write_enc("state.enc.json", state, password)
    print(f"Wrote {len(rows)} holdings, {len(market)} market items; errors: {errors}")
    if not rows:
        sys.exit(1)


if __name__ == "__main__":
    main()
