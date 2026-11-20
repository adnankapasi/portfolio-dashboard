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
    df = df.dropna(subset=["Close"])
    # Yahoo quirks: drop repeated dates and empty placeholder rows (no trades, flat bar).
    df.index = pd.to_datetime(df.index).normalize()
    df = df[~df.index.duplicated(keep="last")]
    if "Volume" in df:
        flat = (df["Volume"].fillna(0) == 0) & (df["High"] == df["Low"])
        df = df[~flat]
    c = df["Close"]
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


def signal(h, ind):
    c = ind["close"]
    if h.get("stop_below") and c < h["stop_below"]:
        return "EXIT ALERT", "red"
    if h.get("add_above") and c > h["add_above"]:
        return "TRIGGER HIT", "green"
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
    data = fetch(tickers)

    rows, errors = [], []
    for h in cfg["holdings"]:
        try:
            ind = indicators(data[h["yahoo"]])
        except Exception as e:  # keep going if one ticker fails
            errors.append(f"{h['symbol']}: {e}")
            continue
        invested = h["qty"] * h["avg"]
        value = h["qty"] * ind["close"]
        sig, color = signal(h, ind)
        print(f"{h['symbol']:<11} {ind['date']} close={ind['close']:.2f} prev={ind['prev_close']:.2f}")
        rows.append({**h, **ind, "invested": invested, "value": value,
                     "pnl": value - invested, "pnl_pct": (value / invested - 1) * 100,
                     "breakeven_pct": (h["avg"] / ind["close"] - 1) * 100,
                     "signal": sig, "color": color})

    macro = []
    for m in cfg["macro"]:
        try:
            ind = indicators(data[m["yahoo"]])
            macro.append({**m, **ind})
        except Exception as e:
            errors.append(f"{m['name']}: {e}")

    inv = sum(r["invested"] for r in rows)
    val = sum(r["value"] for r in rows)
    day = sum(r["qty"] * (r["close"] - r["prev_close"]) for r in rows if r["prev_close"])
    metals = sum(r["value"] for r in rows if r["group"] == "Gold / Silver")
    out = {
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
