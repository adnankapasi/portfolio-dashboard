"""Telegram alerts for the dashboard. Percentages only: no rupee amounts are ever sent.

usage: python scripts/alerts.py evening | morning | weekly | failure | test | chatid
Secrets: TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, DASHBOARD_PASSWORD (and DASHBOARD_URL, optional).
"""
import html
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from update import read_enc, write_enc  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
IST = timezone(timedelta(hours=5, minutes=30))
TOKEN, CHAT = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
URL = os.environ.get("DASHBOARD_URL", "")
e = lambda s: html.escape(str(s), quote=False)
pc = lambda x, d=1: "–" if x is None else f"{'+' if x > 0 else '−' if x < 0 else ''}{abs(x):.{d}f}%"
SENT = []  # for tests


# ------------------------------------------------------------------ telegram
def send(text):
    """Send one message (split if over Telegram's 4096-character limit)."""
    SENT.append(text)
    if not TOKEN or not CHAT:
        print("Telegram not configured (TELEGRAM_TOKEN / TELEGRAM_CHAT_ID). Message would be:\n" + text)
        return False
    import requests
    parts, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) > 3900:
            parts.append(cur); cur = ""
        cur += line + "\n"
    parts.append(cur)
    ok = True
    for p in parts:
        r = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage", timeout=30,
                          json={"chat_id": CHAT, "text": p, "parse_mode": "HTML", "disable_web_page_preview": True})
        if r.status_code != 200:
            print(f"Telegram error {r.status_code}: {r.text[:300]}"); ok = False
    return ok


def footer():
    return f'\n<a href="{e(URL)}">Open dashboard</a>' if URL else ""


# ------------------------------------------------------------------ holding actions (mirror of the page)
def busdays(a, b):
    try:
        return int(np.busday_count(a, b))
    except Exception:
        return 0


def action(h):
    """Returns (status, text) or (None, None). Swing-tagged holdings use the swing rules."""
    c = h["close"]
    if h.get("swing"):
        atr = h.get("atr") or 0.02 * h["avg"]
        risk = min(max(2 * atr / h["avg"], 0.02), 0.10)
        stop, target, be = h["avg"] * (1 - risk), h["avg"] * 1.075, h["avg"] * 1.04
        days = busdays(h.get("first_seen", h["date"]), h["date"])
        if c >= target: return "book", f"🎯 <b>{e(h['symbol'])}</b> swing: book profit, +7.5% target reached ({pc(h['pnl_pct'])})"
        if c <= stop: return "sw_exit", f"🔴 <b>{e(h['symbol'])}</b> swing: exit, stop hit ({pc(h['pnl_pct'])})"
        if days >= 15 and h["pnl_pct"] < 4: return "time", f"⏱ <b>{e(h['symbol'])}</b> swing: time's up, {days} trading days ({pc(h['pnl_pct'])})"
        if c >= be: return "be", f"🔵 <b>{e(h['symbol'])}</b> swing: move stop to your cost (+{(c / h['avg'] - 1) * 100:.1f}%)"
        return None, None
    if h.get("sig") == "exit": return "exit", f"🔴 <b>{e(h['symbol'])}</b>: closed below its exit level ({pc((c / h['stop_below'] - 1) * 100)})"
    if h.get("sig") == "add": return "add", f"🟢 <b>{e(h['symbol'])}</b>: closed above its add level ({pc((c / h['add_above'] - 1) * 100)})"
    if h.get("add_above") and c < h["add_above"] and (h["add_above"] / c - 1) * 100 <= 2:
        return "near_add", f"↗ <b>{e(h['symbol'])}</b>: {(h['add_above'] / c - 1) * 100:.1f}% below its add level"
    if h.get("stop_below") and c >= h["stop_below"] and (c / h["stop_below"] - 1) * 100 <= 2:
        return "near_exit", f"⚠ <b>{e(h['symbol'])}</b>: only {(c / h['stop_below'] - 1) * 100:.1f}% above its exit level"
    return None, None


# ------------------------------------------------------------------ messages
def evening(d, st, cfg):
    mem = st.setdefault("alerts", {})
    swing = {f"{h.get('broker', 'Kite')}:{h['symbol']}" for h in cfg.get("holdings", []) if h.get("swing")}
    acts = []
    for h in d["holdings"]:
        key = f"{h.get('broker', 'Kite')}:{h['symbol']}"
        h = {**h, "swing": key in swing}
        status, text = action(h)
        if status and mem.get(key) != status:
            acts.append(text + (f" · {e(h.get('broker'))}" if sum(x["symbol"] == h["symbol"] for x in d["holdings"]) > 1 else ""))
        mem[key] = status
    for k in [k for k in mem if k not in {f"{h.get('broker', 'Kite')}:{h['symbol']}" for h in d["holdings"]}]:
        del mem[k]
    if acts:
        send(f"⚡ <b>Action alerts · {e(d['price_date'])}</b>\n" + "\n".join(acts) + footer())

    if st.get("summary_sent") == d["price_date"]:
        return  # summary already sent for this trading day (e.g. a re-run after an import)
    s = d["summary"]
    L = [f"📊 <b>Portfolio · {e(d['price_date'])}</b>", f"Today {pc(s['day_pct'], 2)} · Overall {pc(s['pnl_pct'])}"]
    L += [f"{e(b)}: today {pc(x['day_pct'], 2)} · overall {pc(x['pnl_pct'])}" for b, x in d.get("brokers", {}).items()]
    hs = sorted(d["holdings"], key=lambda h: -h["day_pct"])
    if hs:
        L.append("🏆 " + ", ".join(f"{e(h['symbol'])} {pc(h['day_pct'])}" for h in hs[:3]))
        L.append("📉 " + ", ".join(f"{e(h['symbol'])} {pc(h['day_pct'])}" for h in hs[-3:][::-1]))
    sc = d.get("scanner") or {}
    if sc.get("mood"):
        m = sc["mood"]; prev = st.get("mood")
        L.append(f"\n🧭 Market: <b>{e(m['state'].lower())}</b>" + (f" (was {e(prev.lower())})" if prev and prev != m["state"] else "")
                 + f" · {m['pct_above_ema50']}% of stocks above 50-day avg")
        st["mood"] = m["state"]
        setups = sorted([r for r in sc.get("rows", []) if r.get("setup") and not r.get("avoid")], key=lambda r: -r["score"])
        if setups:
            L.append(f"🔎 <b>New swing setups ({len(setups)})</b> · buy only if tomorrow trades above the level")
            for r in setups[:8]:
                p = r["plan"]
                L.append(f"• {e(r['symbol'])} ({e(r['cap'])}) {e(r['setup'])} · score {r['score']} · above {p['trigger']:,.2f}, "
                         f"stop −{p['risk_pct']:.1f}%, target +7.5%" + (f" ⚠ results in {r['results_days']}d" if r.get("results_days") is not None and 0 <= r["results_days"] <= 20 else ""))
            if m["state"] == "Downtrend":
                L.append("⚠ Market in a downtrend: most setups fail now.")
        else:
            L.append("🔎 No new swing setups today.")
    L += metals_changes(d, st)
    if d.get("errors"):
        L.append(f"\nℹ Some data unavailable today: {e(', '.join(x.split(':')[0] for x in d['errors'][:6]))}")
    L.append("\n<i>Percentages only. Not financial advice.</i>" + footer())
    send("\n".join(L))
    st["summary_sent"] = d["price_date"]
    reminder(d, st, cfg)


def metals_changes(d, st):
    X = d.get("metals") or {}
    if X.get("error") or "gold" not in X:
        return []
    old, new, L = st.get("metal_outlook", {}), {}, []
    for m in ("gold", "silver"):
        for h in X[m]["horizons"][:2]:
            k = f"{m}:{h['horizon']}"; new[k] = h["label"]
            if old.get(k) and old[k] != h["label"]:
                hurt = [x["name"] for x in X[m]["drivers"] if x["effect"] == "hurt"][:3]
                helps = [x["name"] for x in X[m]["drivers"] if x["effect"] == "help"][:3]
                why = ("hurting: " + ", ".join(hurt)) if h["label"] == "Negative" else ("helping: " + ", ".join(helps)) if h["label"] == "Positive" else "drivers mixed"
                L.append(f"🪙 {m.title()} {e(h['horizon'])} signal: {e(old[k])} → <b>{e(h['label'])}</b> ({e(why)})")
    st["metal_outlook"] = new
    lv, prev = {}, st.get("metal_levels", {})
    for mk in d.get("market", []):
        for l in mk.get("levels", []):
            k = f"{mk['name']}:{l['value']}"; above = mk["close"] >= l["value"]; lv[k] = above
            if k in prev and prev[k] != above:
                L.append(f"🪙 {e(mk['name'].split(' ')[0])} crossed {'above' if above else 'below'} {l['value']:,} ({e(l['label'].lower())}), {pc(mk['day_pct'])} today")
    st["metal_levels"] = lv
    g, s = X["gold"], X["silver"]
    L.insert(0, f"\n🪙 Gold {pc(g['day_pct'])} · 1–3m signal <b>{e(g['horizons'][0]['label'])}</b> | Silver {pc(s['day_pct'])} · <b>{e(s['horizons'][0]['label'])}</b>")
    return L


def reminder(d, st, cfg):
    """Central bank buying: World Gold Council reports come out about a month after each quarter."""
    today = datetime.now(IST).date()
    if today.month not in (2, 5, 8, 11) or today.day < 3:
        return
    last = (cfg.get("metals") or {}).get("cb_date", "2000-01-01")
    tag = f"{today.year}-{today.month}"
    if last >= f"{today.year}-{today.month:02d}-01" or st.get("cb_reminded") == tag:
        return
    send("🗓 <b>Quarterly reminder</b>\nThe World Gold Council's Gold Demand Trends report for last quarter should be out. "
         "Enter the central bank buying figure on the dashboard (Market → Gold &amp; silver → Central bank buying)." + footer())
    st["cb_reminded"] = tag


def overnight():
    """Overnight moves for the morning brief (Yahoo). Returns list of (name, pct)."""
    try:
        import yfinance as yf
        T = {"S&P 500": "^GSPC", "Nasdaq": "^IXIC", "Gold": "GC=F", "Silver": "SI=F", "Crude oil": "CL=F", "USD/INR": "INR=X", "US 10y yield": "^TNX"}
        raw = yf.download(list(T.values()), period="7d", interval="1d", group_by="ticker", progress=False)
        out = []
        for n, t in T.items():
            c = raw[t]["Close"].dropna()
            if len(c) >= 2:
                out.append((n, (c.iloc[-1] / c.iloc[-2] - 1) * 100))
        return out
    except Exception as ex:
        print(f"overnight: {ex}")
        return []


def morning(d, st, cfg):
    L = [f"🌅 <b>Morning brief · {datetime.now(IST).strftime('%a %d %b')}</b>"]
    ov = overnight()
    if ov:
        L.append("Overnight: " + " · ".join(f"{e(n)} {pc(x)}" for n, x in ov))
    sc = d.get("scanner") or {}
    pend = [t for t in (sc.get("tracker") or {}).get("pending", [])]
    rows = {r["symbol"]: r for r in sc.get("rows", [])}
    if pend:
        L.append(f"\n🔎 <b>Swing setups to watch today</b> (buy only if price trades above)")
        for t in sorted(pend, key=lambda t: -t.get("score", 0))[:8]:
            L.append(f"• {e(t['symbol'])} {e(t['setup'])}: above {t['trigger']:,.2f} · stop −{t['risk_pct']:.1f}% · target +7.5%")
        if sc.get("mood", {}).get("state") == "Downtrend":
            L.append("⚠ Market in a downtrend: consider skipping new trades.")
    swing = {f"{h.get('broker', 'Kite')}:{h['symbol']}" for h in cfg.get("holdings", []) if h.get("swing")}
    near, sw = [], []
    for h in d["holdings"]:
        key = f"{h.get('broker', 'Kite')}:{h['symbol']}"
        if key in swing:
            dup = sum(x["symbol"] == h["symbol"] for x in d["holdings"]) > 1
            sw.append(f"{e(h['symbol'])}{' (' + e(h.get('broker', 'Kite')) + ')' if dup else ''} {pc(h['pnl_pct'])}")
            continue
        if h.get("add_above") and h["close"] < h["add_above"] and (h["add_above"] / h["close"] - 1) * 100 <= 3:
            near.append(f"{e(h['symbol'])} {(h['add_above'] / h['close'] - 1) * 100:.1f}% below add")
        if h.get("stop_below") and h["close"] >= h["stop_below"] and (h["close"] / h["stop_below"] - 1) * 100 <= 3:
            near.append(f"{e(h['symbol'])} {(h['close'] / h['stop_below'] - 1) * 100:.1f}% above exit")
    if sw: L.append("\n⇄ Swing trades: " + ", ".join(sw))
    if near: L.append("👀 Near levels: " + ", ".join(near))
    held = {h["symbol"] for h in d["holdings"]}
    res = sorted([(r["results_days"], s) for s, r in rows.items() if s in held and r.get("results_days") is not None and 0 <= r["results_days"] <= 7])
    if res:
        L.append("📅 Results this week: " + ", ".join(f"{e(s)} in {n}d" for n, s in res))
    if len(L) == 1 + bool(ov):
        L.append("Nothing needs attention before the open.")
    send("\n".join(L) + footer())


def weekly(d, st, cfg):
    H = d.get("history", [])
    L = [f"📅 <b>Weekly review · week to {e(d['price_date'])}</b>"]
    if len(H) >= 2:
        cut = (date.fromisoformat(d["price_date"]) - timedelta(days=7)).isoformat()
        base = [x for x in H if x["d"] <= cut] or H[:1]
        a, b = base[-1], H[-1]
        wk = ((b["v"] - b["i"]) - (a["v"] - a["i"])) / a["v"] * 100 if a["v"] else 0
        L.append(f"Portfolio this week: {pc(wk, 2)} · overall {pc(d['summary']['pnl_pct'])}")
    ch, moves = d.get("charts", {}), {}
    for h in d["holdings"]:
        c = (ch.get(h["symbol"]) or {}).get("c") or []
        if len(c) >= 6 and c[-6] and c[-1]:
            moves[h["symbol"]] = (c[-1] / c[-6] - 1) * 100  # last 5 trading days
    if moves:
        wkchg = sorted(moves.items(), key=lambda x: -x[1])
        L.append("🏆 " + ", ".join(f"{e(s)} {pc(x)}" for s, x in wkchg[:3]) + "\n📉 " + ", ".join(f"{e(s)} {pc(x)}" for s, x in wkchg[-3:][::-1]))
    T = ((d.get("scanner") or {}).get("tracker") or {})
    o = T.get("overall") or {}
    if o.get("trades"):
        L.append(f"\n🔎 Scanner track record since {e(T.get('since'))}: {o['trades']} closed · win {o['win_pct']}% · avg {o['avg_r']:+.2f}R per trade")
        L.append("   " + " · ".join(f"{e(k)} {v['avg_r']:+.2f}R ({v['trades']})" for k, v in T.get("by_setup", {}).items() if v.get("trades")))
    else:
        L.append(f"\n🔎 Scanner: {T.get('signals', 0)} signals tracked so far, none closed yet.")
    X = d.get("metals") or {}
    if "gold" in X:
        for m in ("gold", "silver"):
            L.append(f"🪙 {m.title()} {pc(X[m]['chg_1m'])} 1 month · signals: " + ", ".join(f"{e(h['horizon'])} {e(h['label'])}" for h in X[m]["horizons"]))
    L.append("\n<i>Percentages only. Not financial advice.</i>" + footer())
    send("\n".join(L))


def chatid():
    import requests
    if not TOKEN:
        print("Add the TELEGRAM_TOKEN secret first."); return
    r = requests.get(f"https://api.telegram.org/bot{TOKEN}/getUpdates", timeout=30).json()
    ids = {(u.get("message") or {}).get("chat", {}).get("id"): (u.get("message") or {}).get("chat", {}).get("first_name") for u in r.get("result", [])}
    ids.pop(None, None)
    if not ids:
        print("No messages found. Open your bot in Telegram, send it 'hi', then run this again.")
    for i, n in ids.items():
        print(f"Chat ID: {i}   (from {n})  -> save this as the TELEGRAM_CHAT_ID secret")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "evening"
    if mode == "chatid":
        return chatid()
    if mode == "test":
        ok = send("✅ <b>Dashboard alerts are connected.</b>\nYou'll get the morning brief at 8:45 AM, action alerts and the daily summary after 4:45 PM, and a weekly review on Saturdays." + footer())
        sys.exit(0 if ok else 1)
    if mode == "failure":
        send("❗ <b>Dashboard update failed today.</b>\nNumbers on the dashboard may be out of date. Check the latest run under Actions on GitHub." + footer())
        return
    pw = os.environ.get("DASHBOARD_PASSWORD")
    d, st, cfg = read_enc("data.enc.json", pw), read_enc("state.enc.json", pw, {}) or {}, read_enc("holdings.enc.json", pw, {}) or {}
    if d is None:
        print("No dashboard data yet."); return
    {"evening": evening, "morning": morning, "weekly": weekly}[mode](d, st, cfg)
    if mode == "evening":  # only the evening run remembers what was sent (avoids repeat alerts)
        write_enc("state.enc.json", st, pw)


if __name__ == "__main__":
    main()
