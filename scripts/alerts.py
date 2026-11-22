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
    text = text.replace("</pre>\n\n", "</pre>\n")  # a table already ends its own line
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
    return f'\n\n🔗 <a href="{e(URL)}">Open dashboard</a>' if URL else ""


def nice_date(iso):
    try:
        return date.fromisoformat(iso).strftime("%a, %d %b %Y")
    except Exception:
        return iso


def head(emoji, title, sub):
    return f"{emoji} <b>{e(title)}</b>\n<i>{e(sub)}</i>"


def table(rows, header=None):
    """Monospaced, aligned table (Telegram <pre>). First column left, others right-aligned."""
    allr = ([header] if header else []) + rows
    w = [max(len(str(r[i])) for r in allr) for i in range(len(allr[0]))]
    fmt = lambda r: "  ".join(str(c).ljust(w[i]) if i == 0 else str(c).rjust(w[i]) for i, c in enumerate(r)).rstrip()
    return "<pre>" + "\n".join(e(fmt(r)) for r in allr) + "</pre>"


def section(emoji, title):
    return f"\n\n{emoji} <b>{e(title)}</b>"


# ------------------------------------------------------------------ holding actions (mirror of the page)
def busdays(a, b):
    try:
        return int(np.busday_count(a, b))
    except Exception:
        return 0


GROUPS = [("exit", "🔴", "Exit"), ("sw_exit", "🔴", "Swing: exit (stop hit)"), ("book", "🎯", "Swing: book profit"),
          ("time", "⏱", "Swing: time's up"), ("be", "🔵", "Swing: move stop to your cost"), ("add", "🟢", "Add zone"),
          ("near_exit", "⚠️", "Close to exit level"), ("near_add", "↗️", "Close to add level")]


def action(h):
    """Returns (status, detail) or (None, None). Swing-tagged holdings use the swing rules."""
    c = h["close"]
    if h.get("swing"):
        atr = h.get("atr") or 0.02 * h["avg"]
        risk = min(max(2 * atr / h["avg"], 0.02), 0.10)
        stop, target, be = h["avg"] * (1 - risk), h["avg"] * 1.075, h["avg"] * 1.04
        days = busdays(h.get("first_seen", h["date"]), h["date"])
        if c >= target: return "book", f"+7.5% target reached · now {pc(h['pnl_pct'])}"
        if c <= stop: return "sw_exit", f"stop hit · now {pc(h['pnl_pct'])}"
        if days >= 15 and h["pnl_pct"] < 4: return "time", f"{days} trading days without a move · now {pc(h['pnl_pct'])}"
        if c >= be: return "be", f"up {pc((c / h['avg'] - 1) * 100)}: protect it, move stop to your cost"
        return None, None
    if h.get("sig") == "exit": return "exit", f"closed {abs((c / h['stop_below'] - 1) * 100):.1f}% below its exit level"
    if h.get("sig") == "add": return "add", f"closed {(c / h['add_above'] - 1) * 100:.1f}% above its add level"
    if h.get("add_above") and c < h["add_above"] and (h["add_above"] / c - 1) * 100 <= 2:
        return "near_add", f"{(h['add_above'] / c - 1) * 100:.1f}% below its add level"
    if h.get("stop_below") and c >= h["stop_below"] and (c / h["stop_below"] - 1) * 100 <= 2:
        return "near_exit", f"only {(c / h['stop_below'] - 1) * 100:.1f}% above its exit level"
    return None, None


def setup_lines(items, target=7.5):
    out = []
    for r in items:
        res = f"\n   ⚠️ results in {r['res']} days" if r.get("res") is not None and 0 <= r["res"] <= 20 else ""
        out.append(f"\n• <b>{e(r['symbol'])}</b> · {e(r['setup'])}{' · ' + e(r['cap']) + ' cap' if r.get('cap') else ''}{' · score ' + str(r['score']) if r.get('score') is not None else ''}"
                   f"\n   Buy above <b>{r['trigger']:,.2f}</b> · stop −{r['risk_pct']:.1f}% · target +{target:g}%{res}")
    return "".join(out)


# ------------------------------------------------------------------ messages
def evening(d, st, cfg):
    mem = st.setdefault("alerts", {})
    swing = {f"{h.get('broker', 'Kite')}:{h['symbol']}" for h in cfg.get("holdings", []) if h.get("swing")}
    found = {}
    for h in d["holdings"]:
        key = f"{h.get('broker', 'Kite')}:{h['symbol']}"
        h = {**h, "swing": key in swing}
        status, detail = action(h)
        if status and mem.get(key) != status:
            dup = sum(x["symbol"] == h["symbol"] for x in d["holdings"]) > 1
            found.setdefault(status, []).append(f"• <b>{e(h['symbol'])}</b>{' (' + e(h.get('broker', 'Kite')) + ')' if dup else ''}: {e(detail)}")
        mem[key] = status
    held = {f"{h.get('broker', 'Kite')}:{h['symbol']}" for h in d["holdings"]}
    for k in [k for k in mem if k not in held]:
        del mem[k]
    if found:
        msg = head("⚡", "Action alerts", f"{nice_date(d['price_date'])} · after market close")
        for status, emo, title in GROUPS:
            if status in found:
                msg += section(emo, title) + "\n" + "\n".join(found[status])
        send(msg + footer())

    if st.get("summary_sent") == d["price_date"]:
        return  # summary already sent for this trading day (e.g. a re-run after an import)
    s = d["summary"]
    msg = head("📊", "Daily summary", nice_date(d["price_date"]))
    rows = [("All", pc(s["day_pct"], 2), pc(s["pnl_pct"]))] + [(b, pc(x["day_pct"], 2), pc(x["pnl_pct"])) for b, x in d.get("brokers", {}).items()]
    msg += section("💼", "Portfolio") + "\n" + table(rows, ("", "Today", "Overall"))
    best = {}
    for h in d["holdings"]:
        best[h["symbol"]] = h["day_pct"]
    hs = sorted(best.items(), key=lambda x: -x[1])
    if hs:
        msg += section("📈", "Top gainers today") + "\n" + table([(s_, pc(x)) for s_, x in hs[:3]])
        msg += section("📉", "Top losers today") + "\n" + table([(s_, pc(x)) for s_, x in hs[-3:][::-1]])
    sc = d.get("scanner") or {}
    if sc.get("mood"):
        m = sc["mood"]; prev = st.get("mood")
        tip = {"Uptrend": "Good conditions for new swing trades.", "Choppy": "Breakouts fail more often: be selective and size smaller.",
               "Downtrend": "Most setups fail now: consider no new trades."}[m["state"]]
        msg += section("🧭", f"Market mood: {m['state']}" + (f" (was {prev})" if prev and prev != m["state"] else ""))
        msg += f"\n{m['pct_above_ema50']:.0f}% of stocks above their 50-day average.\n<i>{tip}</i>"
        st["mood"] = m["state"]
        setups = sorted([r for r in sc.get("rows", []) if r.get("setup") and not r.get("avoid")], key=lambda r: -r["score"])
        if setups:
            msg += section("🔎", f"New swing setups ({len(setups)})") + "\n<i>Buy only if the price trades above the level tomorrow.</i>"
            msg += setup_lines([{**r, **r["plan"], "res": r.get("results_days")} for r in setups[:6]])
            if len(setups) > 6:
                msg += f"\n<i>+{len(setups) - 6} more on the dashboard</i>"
        else:
            msg += section("🔎", "New swing setups") + "\nNone today. That's normal: good setups are rare."
    msg += etf_block(d, st)
    msg += metals_block(d, st)
    if d.get("errors"):
        msg += f"\n\nℹ️ <i>Data unavailable today: {e(', '.join(x.split(':')[0] for x in d['errors'][:6]))}</i>"
    msg += "\n\n<i>Percentages only · not financial advice</i>" + footer()
    send(msg)
    st["summary_sent"] = d["price_date"]
    reminder(d, st, cfg)


def etf_block(d, st):
    E = d.get("etfs") or {}
    if E.get("error") or not E.get("themes"):
        return ""
    out = section("🧺", "ETF themes") + f"\nLeading: {e(', '.join(E['leading']))}\nLagging: {e(', '.join(E['lagging']))}"
    old, new, ch = st.get("etf_sig", {}), {}, []
    for t in E["themes"]:
        new[t["name"]] = t["s612"]
        if old.get(t["name"]) and old[t["name"]] != t["s612"]:
            ch.append(f"• {e(t['name'])}: {e(old[t['name']])} → <b>{e(t['s612'])}</b>")
    st["etf_sig"] = new
    if ch:
        out += "\n🔔 <b>6–12 month signal changed</b>\n" + "\n".join(ch)
    ok = sorted([r for r in E.get("swing", []) if not r["avoid"]], key=lambda r: -r["score"])
    R = E.get("rules", {})
    if ok:
        out += f"\n\n<b>ETF swing setups ({len(ok)})</b>" + setup_lines(
            [{"symbol": r["symbol"], "setup": r["setup"], "cap": None, "score": r["score"], **r["plan"]} for r in ok[:4]], R.get("target_pct", 5))
    return out


def metals_block(d, st):
    X = d.get("metals") or {}
    if X.get("error") or "gold" not in X:
        return ""
    old, new, changes = st.get("metal_outlook", {}), {}, []
    for m in ("gold", "silver"):
        for h in X[m]["horizons"][:2]:
            k = f"{m}:{h['horizon']}"; new[k] = h["label"]
            if old.get(k) and old[k] != h["label"]:
                hurt = [x["name"] for x in X[m]["drivers"] if x["effect"] == "hurt"][:3]
                helps = [x["name"] for x in X[m]["drivers"] if x["effect"] == "help"][:3]
                why = ("hurting: " + ", ".join(hurt)) if h["label"] == "Negative" else ("helping: " + ", ".join(helps)) if h["label"] == "Positive" else "drivers mixed"
                changes.append(f"• {m.title()} {e(h['horizon'])}: {e(old[k])} → <b>{e(h['label'])}</b>\n   <i>{e(why)}</i>")
    st["metal_outlook"] = new
    lv, prev = {}, st.get("metal_levels", {})
    for mk in d.get("market", []):
        for l in mk.get("levels", []):
            k = f"{mk['name']}:{l['value']}"; above = mk["close"] >= l["value"]; lv[k] = above
            if k in prev and prev[k] != above:
                changes.append(f"• {e(mk['name'].split(' ')[0])} crossed <b>{'above' if above else 'below'} {l['value']:,}</b> ({e(l['label'].lower())})")
    st["metal_levels"] = lv
    g, s = X["gold"], X["silver"]
    out = section("🪙", "Gold & silver") + "\n" + table([("Gold", pc(g["day_pct"]), g["horizons"][0]["label"], g["horizons"][1]["label"]),
                                                         ("Silver", pc(s["day_pct"]), s["horizons"][0]["label"], s["horizons"][1]["label"])],
                                                        ("", "Today", "1–3m", "6–12m"))
    if changes:
        out += "\n🔔 <b>Changed today</b>\n" + "\n".join(changes)
    return out


def reminder(d, st, cfg):
    """Central bank buying: World Gold Council reports come out about a month after each quarter."""
    today = datetime.now(IST).date()
    if today.month not in (2, 5, 8, 11) or today.day < 3:
        return
    last = (cfg.get("metals") or {}).get("cb_date", "2000-01-01")
    tag = f"{today.year}-{today.month}"
    if last >= f"{today.year}-{today.month:02d}-01" or st.get("cb_reminded") == tag:
        return
    send(head("🗓", "Quarterly reminder", "Central bank gold buying")
         + "\n\nThe World Gold Council's <b>Gold Demand Trends</b> report for last quarter should be out now."
         + "\n\nEnter the figure on the dashboard:\n<i>Market → Gold &amp; silver → Central bank buying</i>" + footer())
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
    msg = head("🌅", "Morning brief", f"{datetime.now(IST).strftime('%a, %d %b %Y')} · market opens 9:15 AM")
    ov = overnight()
    if ov:
        msg += section("🌍", "Overnight") + "\n" + table([(n, pc(x)) for n, x in ov])
    sc = d.get("scanner") or {}
    pend = sorted((sc.get("tracker") or {}).get("pending", []), key=lambda t: -t.get("score", 0))
    rows = {r["symbol"]: r for r in sc.get("rows", [])}
    quiet = True
    if pend:
        quiet = False
        msg += section("🔎", f"Swing setups to watch ({len(pend)})") + "\n<i>Buy only if the price trades above the level today.</i>"
        msg += setup_lines([{**t, "cap": rows.get(t["symbol"], {}).get("cap"), "res": rows.get(t["symbol"], {}).get("results_days")} for t in pend[:6]])
        if sc.get("mood", {}).get("state") == "Downtrend":
            msg += "\n\n⚠️ <i>Market is in a downtrend: consider skipping new trades.</i>"
    E = d.get("etfs") or {}
    ep = sorted(((E.get("tracker") or {}).get("pending") or []), key=lambda t: -t.get("score", 0))
    if ep:
        quiet = False
        msg += section("🧺", f"ETF setups to watch ({len(ep)})") + "\n<i>Same rule: buy only above the level.</i>"
        msg += setup_lines([{**t, "cap": None, "res": None} for t in ep[:4]], (E.get("rules") or {}).get("target_pct", 5))
    swing = {f"{h.get('broker', 'Kite')}:{h['symbol']}" for h in cfg.get("holdings", []) if h.get("swing")}
    near, sw = [], []
    for h in d["holdings"]:
        key = f"{h.get('broker', 'Kite')}:{h['symbol']}"
        dup = sum(x["symbol"] == h["symbol"] for x in d["holdings"]) > 1
        name = h["symbol"] + (f" ({h.get('broker', 'Kite')})" if dup else "")
        if key in swing:
            sw.append((name, pc(h["pnl_pct"])))
            continue
        if h.get("add_above") and h["close"] < h["add_above"] and (h["add_above"] / h["close"] - 1) * 100 <= 3:
            near.append(f"• <b>{e(name)}</b>: {(h['add_above'] / h['close'] - 1) * 100:.1f}% below its add level")
        if h.get("stop_below") and h["close"] >= h["stop_below"] and (h["close"] / h["stop_below"] - 1) * 100 <= 3:
            near.append(f"• <b>{e(name)}</b>: {(h['close'] / h['stop_below'] - 1) * 100:.1f}% above its exit level")
    if sw:
        quiet = False
        msg += section("⇄", "Your swing trades") + "\n" + table(sw)
    if near:
        quiet = False
        msg += section("👀", "Holdings near their levels") + "\n" + "\n".join(near)
    held = {h["symbol"] for h in d["holdings"]}
    res = sorted([(r["results_days"], s) for s, r in rows.items() if s in held and r.get("results_days") is not None and 0 <= r["results_days"] <= 7])
    if res:
        quiet = False
        msg += section("📅", "Results this week") + "\n" + "\n".join(f"• <b>{e(s)}</b> in {n} day{'s' if n != 1 else ''}" for n, s in res)
    if quiet:
        msg += "\n\n✅ Nothing needs your attention before the open."
    send(msg + footer())


def weekly(d, st, cfg):
    msg = head("📅", "Weekly review", f"Week to {nice_date(d['price_date'])}")
    H = d.get("history", [])
    rows = []
    if len(H) >= 2:
        cut = (date.fromisoformat(d["price_date"]) - timedelta(days=7)).isoformat()
        base = [x for x in H if x["d"] <= cut] or H[:1]
        a, b = base[-1], H[-1]
        rows.append(("This week", pc(((b["v"] - b["i"]) - (a["v"] - a["i"])) / a["v"] * 100 if a["v"] else 0, 2)))
    rows.append(("Overall", pc(d["summary"]["pnl_pct"])))
    msg += section("💼", "Portfolio") + "\n" + table(rows)
    ch, moves = d.get("charts", {}), {}
    for h in d["holdings"]:
        c = (ch.get(h["symbol"]) or {}).get("c") or []
        if len(c) >= 6 and c[-6] and c[-1]:
            moves[h["symbol"]] = (c[-1] / c[-6] - 1) * 100  # last 5 trading days
    if moves:
        wk = sorted(moves.items(), key=lambda x: -x[1])
        msg += section("📈", "Best this week") + "\n" + table([(s_, pc(x)) for s_, x in wk[:3]])
        msg += section("📉", "Worst this week") + "\n" + table([(s_, pc(x)) for s_, x in wk[-3:][::-1]])
    T = ((d.get("scanner") or {}).get("tracker") or {})
    o = T.get("overall") or {}
    msg += section("🔎", "Scanner track record")
    if o.get("trades"):
        msg += f"\n<i>Since {e(nice_date(T.get('since')))}</i>\n" + table(
            [(k, str(v["trades"]), f"{v['win_pct']:.0f}%", f"{v['avg_r']:+.2f}R") for k, v in T.get("by_setup", {}).items() if v.get("trades")]
            + [("All setups", str(o["trades"]), f"{o['win_pct']:.0f}%", f"{o['avg_r']:+.2f}R")], ("", "Trades", "Win", "Avg"))
        msg += "\n<i>R = your risk per trade. Judge after 30–40 closed trades.</i>"
    else:
        msg += f"\n{T.get('signals', 0)} signals tracked so far, none closed yet."
    X = d.get("metals") or {}
    if "gold" in X:
        short = {"1–3 months": "1–3m", "6–12 months": "6–12m", "1–2 years": "1–2y", "3–5 years": "3–5y"}
        msg += section("🪙", "Gold & silver")
        for m in ("gold", "silver"):
            hz = [f"{short[h['horizon']]} {e(h['label'])}" for h in X[m]["horizons"]]
            msg += f"\n<b>{m.title()}</b> {pc(X[m]['chg_1m'])} this month\n   " + " · ".join(hz[:2]) + "\n   " + " · ".join(hz[2:])
    msg += "\n\n<i>Percentages only · not financial advice</i>" + footer()
    send(msg)


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
        ok = send(head("✅", "Dashboard alerts are connected", "You're all set")
                  + "\n\n<b>What you'll get</b>\n• 🌅 Morning brief · 8:45 AM, Mon–Fri\n• ⚡ Action alerts · after 4:45 PM, only when something changes"
                  + "\n• 📊 Daily summary · after 4:45 PM\n• 📅 Weekly review · Saturday 9 AM" + footer())
        sys.exit(0 if ok else 1)
    if mode == "failure":
        send(head("❗", "Dashboard update failed", datetime.now(IST).strftime("%a, %d %b %Y"))
             + "\n\nNumbers on the dashboard may be out of date.\nCheck the latest run under <b>Actions</b> on GitHub." + footer())
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
