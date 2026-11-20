"""Fetch daily prices, compute indicators and signals, write data.json."""
import base64
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IST = timezone(timedelta(hours=5, minutes=30))


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
    pdm = np.where((upm > dnm) & (upm > 0), upm, 0.0)
    ndm = np.where((dnm > upm) & (dnm > 0), dnm, 0.0)
    a = atr(df, n)
    pdi = 100 * pd.Series(pdm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / a
    ndi = 100 * pd.Series(ndm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / a
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
    level = np.where(direction == 1, fl, fu)
    return int(direction[-1]), float(level[-1])


def indicators(df):
    if df is None or df.dropna(subset=["Close"]).empty:
        raise ValueError("no price data from NSE, BSE or Yahoo")
    df = df.dropna(subset=["Close"])
    # Yahoo quirks: drop repeated dates and empty placeholder rows (no trades, flat bar).
    df.index = pd.to_datetime(df.index).normalize()
    df = df[~df.index.duplicated(keep="last")]
    if "Volume" in df:
        flat = (df["Volume"].fillna(0) == 0) & (df["High"] == df["Low"])
        df = df[~flat]
    c = df["Close"]
    if len(c) < 15:  # too little history for indicators (e.g. new listing)
        return {"close": float(c.iloc[-1]), "prev_close": float(c.iloc[-2]) if len(c) > 1 else None,
                "date": df.index[-1].strftime("%Y-%m-%d"), "ema50": None, "sma200": None, "rsi": None,
                "adx": None, "st_dir": 0, "st_level": None, "spark": [round(float(x), 2) for x in c.tail(60)]}
    d, lvl = supertrend(df)
    out = {
        "close": float(c.iloc[-1]),
        "prev_close": float(c.iloc[-2]) if len(c) > 1 else None,
        "date": df.index[-1].strftime("%Y-%m-%d"),
        "ema50": float(c.ewm(span=50, adjust=False).mean().iloc[-1]),
        "sma200": float(c.rolling(200).mean().iloc[-1]) if len(c) >= 200 else None,
        "rsi": float(rsi(c).iloc[-1]),
        "adx": float(adx(df).iloc[-1]),
        "st_dir": d,
        "st_level": lvl,
        "spark": [round(float(x), 2) for x in c.tail(60)],
    }
    return out


def auto_levels(h, ind):
    """Same rules used for the Kite stocks, applied daily to holdings without custom levels."""
    c, ema, st, d = ind["close"], ind["ema50"], ind["st_level"], ind["st_dir"]
    if d == 0 or ema is None:
        return h
    h = {**h, "auto": True}
    if d == -1:   # downtrend: don't add until the Supertrend flips
        h["add_above"], h["stop_below"] = round(st, 2), None
        h["note"] = f"Downtrend. Don't add until a close above {st:,.2f} (Supertrend flip)."
    elif c < ema:  # trend turning up but still under the 50-EMA
        h["add_above"], h["stop_below"] = round(ema, 2), round(st, 2)
        h["note"] = f"Recovering. Add only after a close above the 50-EMA ({ema:,.2f}); exit below {st:,.2f}."
    else:          # confirmed uptrend
        h["add_above"], h["stop_below"] = None, round(st, 2)
        h["note"] = f"Uptrend confirmed. Hold; trail the stop at the Supertrend ({st:,.2f})."
    return h


def signal(h, ind):
    if h.get("auto") and ind["st_dir"] == 1 and h.get("add_above") is None and ind["close"] >= h["stop_below"]:
        return "HOLD · uptrend", "green"
    if h.get("auto") and ind["st_dir"] == -1:
        return "DON'T ADD · downtrend", "grey"
    c = ind["close"]
    if h.get("stop_below") and c < h["stop_below"]:
        return "EXIT ALERT", "red"
    if h.get("add_above") and c > h["add_above"]:
        return "TRIGGER HIT", "green"
    if ind["st_dir"] == 0:
        return "WAIT · no trend data", "grey"
    if ind["st_dir"] == 1:
        return "WAIT · trend up", "amber"
    return "WAIT · trend down", "grey"


def encrypt(text, password):
    """AES-256-GCM with a PBKDF2-SHA256 key; matches the WebCrypto code in index.html."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    salt, iv, iters = os.urandom(16), os.urandom(12), 310000
    key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=iters).derive(password.encode())
    ct = AESGCM(key).encrypt(iv, text.encode(), None)
    b64 = lambda b: base64.b64encode(b).decode()
    return {"v": 1, "iter": iters, "salt": b64(salt), "iv": b64(iv), "ct": b64(ct)}


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
    """Official end-of-day file (UDiFF format). Returns (date, {symbol: row}) or (None, {})."""
    import io
    import zipfile
    import requests
    home, url = BHAV_URLS[exchange]
    sess = requests.Session()
    sess.headers.update({**HEADERS, "Referer": home})
    try:
        sess.get(home, timeout=15)  # pick up cookies
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
        rows = {}
        for _, r_ in df.iterrows():
            sym = str(r_.get("TckrSymb", "")).strip()
            ser = str(r_.get("SctySrs", "")).strip()
            rank = SERIES_PREF.index(ser) if ser in SERIES_PREF else 99
            if sym and (sym not in rows or rank < rows[sym]["_rank"]):
                rows[sym] = {"_rank": rank, "open": float(r_["OpnPric"]), "high": float(r_["HghPric"]),
                             "low": float(r_["LwPric"]), "close": float(r_["ClsPric"]),
                             "prev": float(r_["PrvsClsgPric"])}
        print(f"{exchange} bhavcopy {day}: {len(rows)} symbols")
        return pd.Timestamp(day), rows
    return None, {}


def merge_official(hist, date, row):
    """Put the exchange's official OHLC for `date` onto the Yahoo history."""
    new = pd.DataFrame({"Open": [row["open"]], "High": [row["high"]], "Low": [row["low"]],
                        "Close": [row["close"]], "Volume": [1.0]}, index=[date])
    if hist is None or hist.dropna(subset=["Close"]).empty:
        prev = pd.DataFrame({"Open": [row["prev"]], "High": [row["prev"]], "Low": [row["prev"]],
                             "Close": [row["prev"]], "Volume": [1.0]}, index=[date - pd.Timedelta(days=1)])
        return pd.concat([prev, new])
    hist = hist.dropna(subset=["Close"]).copy()
    hist.index = pd.to_datetime(hist.index).normalize()
    hist = hist[hist.index < date]
    if len(hist):  # make yesterday match the exchange's official previous close
        hist.iloc[-1, hist.columns.get_loc("Close")] = row["prev"]
    return pd.concat([hist[["Open", "High", "Low", "Close", "Volume"]], new])


def fetch(tickers):
    import yfinance as yf
    raw = yf.download(tickers, period="2y", interval="1d", group_by="ticker",
                      auto_adjust=False, progress=False, threads=True)
    return {t: raw[t] for t in tickers if t in raw.columns.get_level_values(0)}


def main():
    # Holdings come from the HOLDINGS_JSON secret on GitHub; local file only for testing.
    raw_cfg = os.environ.get("HOLDINGS_JSON") or (ROOT / "holdings.json").read_text()
    cfg = json.loads(raw_cfg)
    tickers = [h["yahoo"] for h in cfg["holdings"]] + [m["yahoo"] for m in cfg["macro"]]
    try:
        data = fetch(tickers)
    except Exception as e:
        print(f"Yahoo failed: {e}")
        data = {}
    bhav = {ex: fetch_bhavcopy(ex) for ex in ("NSE", "BSE")}

    rows, errors = [], []
    for h in cfg["holdings"]:
        sym = h.get("nse", h["symbol"])
        hist, source = data.get(h["yahoo"]), "Yahoo"
        for ex in ("NSE", "BSE"):
            bdate, brows = bhav[ex]
            key = sym if ex == "NSE" else h.get("bse", sym)
            if bdate is not None and key in brows:
                hist, source = merge_official(hist, bdate, brows[key]), ex
                break
        try:
            ind = indicators(hist)
            ind["source"] = source
            if h.get("add_above") is None and h.get("stop_below") is None and h["group"] != "Gold / Silver":
                h = auto_levels(h, ind)
        except Exception as e:  # keep going if one ticker fails
            errors.append(f"{h['symbol']}: {e}")
            continue
        invested = h["qty"] * h["avg"]
        value = h["qty"] * ind["close"]
        sig, color = signal(h, ind)
        print(f"{h['symbol']:<11} {ind['date']} close={ind['close']:.2f} prev={ind['prev_close']} [{source}]")
        rows.append({**h, **ind, "invested": invested, "value": value,
                     "pnl": value - invested, "pnl_pct": (value / invested - 1) * 100,
                     "breakeven_pct": (h["avg"] / ind["close"] - 1) * 100,
                     "signal": sig, "color": color})

    macro = []
    for m in cfg["macro"]:
        try:
            ind = indicators(data.get(m["yahoo"]))
            macro.append({**m, **ind})
        except Exception as e:
            errors.append(f"{m['name']}: {e}")

    inv = sum(r["invested"] for r in rows)
    val = sum(r["value"] for r in rows)
    day = sum(r["qty"] * (r["close"] - r["prev_close"]) for r in rows if r["prev_close"])
    metals = sum(r["value"] for r in rows if r["group"] == "Gold / Silver")
    brokers = {}
    for r in rows:
        b = brokers.setdefault(r.get("broker", "Kite"), {"invested": 0, "value": 0, "day": 0})
        b["invested"] += r["invested"]; b["value"] += r["value"]
        if r["prev_close"]:
            b["day"] += r["qty"] * (r["close"] - r["prev_close"])
    for b in brokers.values():
        b["pnl"] = b["value"] - b["invested"]
        b["pnl_pct"] = (b["value"] / b["invested"] - 1) * 100 if b["invested"] else 0
    out = {
        "sources": {ex: (bhav[ex][0].strftime("%Y-%m-%d") if bhav[ex][0] is not None else None) for ex in bhav},
        "brokers": brokers,
        "updated": datetime.now(IST).strftime("%d %b %Y, %I:%M %p IST"),
        "summary": {"invested": inv, "value": val, "pnl": val - inv,
                    "pnl_pct": (val / inv - 1) * 100 if inv else 0, "day": day,
                    "metals_pct": metals / val * 100 if val else 0},
        "holdings": rows, "macro": macro, "errors": errors,
    }
    password = os.environ.get("DASHBOARD_PASSWORD")
    if not password:
        sys.exit("DASHBOARD_PASSWORD is not set - refusing to publish unencrypted data.")
    (ROOT / "data.enc.json").write_text(json.dumps(encrypt(json.dumps(out, default=float), password)))
    print(f"Wrote {len(rows)} holdings, {len(macro)} macro; errors: {errors}")
    if not rows:
        sys.exit(1)


if __name__ == "__main__":
    main()
