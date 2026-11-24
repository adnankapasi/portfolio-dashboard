"""IPO last-day check (no GMP — official NSE data only).

Runs at 3 PM IST Mon–Fri (cron-job.org -> alerts.yml mode=ipo). If an IPO's bidding closes today it sends:
  subscription by category (QIB / HNI / retail) · market mood · how recent IPOs listed · a verdict.
It also logs every alerted IPO and, once listed, records its listing-day result, so the rule
gets checked on real outcomes over time ("forward tracking").

Percentages and multiples only: nothing about amounts.
"""
import re
from datetime import date, datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))
NSE = "https://www.nseindia.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
           "Accept": "application/json,text/html,*/*", "Accept-Language": "en-US,en;q=0.9"}

# Verdict rule (starting point — forward tracking will show whether it holds)
STRONG_QIB, STRONG_HNI, WEAK_QIB = 25, 25, 5


# ------------------------------------------------------------------ NSE
def nse_session():
    import requests
    s = requests.Session(); s.headers.update({**HEADERS, "Referer": f"{NSE}/market-data/all-upcoming-issues-ipo"})
    s.get(NSE + "/", timeout=15)
    return s


def get_json(s, path):
    try:
        r = s.get(NSE + path, timeout=20)
        if r.status_code != 200:
            print(f"NSE {path}: HTTP {r.status_code}"); return None
        return r.json()
    except Exception as ex:
        print(f"NSE {path}: {ex}"); return None


def pdate(x):
    for f in ("%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d", "%d %b %Y"):
        try:
            return datetime.strptime(str(x).strip(), f).date()
        except Exception:
            pass
    return None


def num(x):
    try:
        return float(re.sub(r"[^0-9.]", "", str(x)) or "nan")
    except Exception:
        return None


def price_band(txt):
    """'Rs.100 to Rs.105' -> 105.0 (upper band)."""
    v = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", str(txt or ""))]
    return max(v) if v else None


def categories(data):
    """NSE category table -> {'qib': x, 'hni': x, 'bhni': x, 'shni': x, 'retail': x, 'total': x} (times subscribed)."""
    rows = data.get("dataList") if isinstance(data, dict) else data
    out = {}
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        cat = str(r.get("category") or r.get("srNo") or "").lower()
        t = None
        for k, v in r.items():
            if "time" in k.lower():
                t = num(v); break
        if t is None or t != t:
            continue
        if "qualified" in cat or "qib" in cat:
            out.setdefault("qib", t)
        elif "more than ten" in cat or "10 lakh" in cat and "more" in cat or "bnii" in cat or "big" in cat:
            out["bhni"] = t
        elif "two lakh" in cat or "2 lakh" in cat and "ten" in cat or "snii" in cat or "small" in cat:
            out["shni"] = t
        elif "non institutional" in cat or "non-institutional" in cat or "nii" in cat or "hni" in cat:
            out.setdefault("hni", t)
        elif "retail" in cat or "rii" in cat:
            out.setdefault("retail", t)
        elif cat.startswith("total"):
            out["total"] = t
    return out


def closing_today(s, today):
    cur = get_json(s, "/api/ipo-current-issue") or []
    if isinstance(cur, dict):
        cur = cur.get("data") or []
    out = []
    for x in cur:
        if pdate(x.get("issueEndDate")) == today:
            out.append({"symbol": x.get("symbol"), "name": x.get("companyName") or x.get("symbol"),
                        "sme": str(x.get("series", "")).upper() == "SME" or "sme" in str(x.get("securityType", "")).lower(),
                        "price": price_band(x.get("issuePrice")), "total": num(x.get("noOfTime"))})
    return out


# ------------------------------------------------------------------ listing results
def listing_gain(symbol, issue_price, sme=False):
    """Listing-day open and close vs issue price, from Yahoo (NSE mainboard only; SME isn't on Yahoo reliably)."""
    if not issue_price or sme:
        return None
    try:
        import yfinance as yf
        h = yf.Ticker(f"{symbol}.NS").history(period="1y", interval="1d", auto_adjust=False)
        if h is None or h.empty:
            return None
        first = h.iloc[0]
        return {"date": h.index[0].strftime("%Y-%m-%d"),
                "open": round((first["Open"] / issue_price - 1) * 100, 1),
                "close": round((first["Close"] / issue_price - 1) * 100, 1)}
    except Exception as ex:
        print(f"listing {symbol}: {ex}"); return None


def recent_listings(s, n=10):
    """Last n mainboard IPOs: how they opened on listing day vs issue price."""
    past = get_json(s, "/api/public-past-issues") or []
    if isinstance(past, dict):
        past = past.get("data") or []
    rows = []
    for x in past:
        if str(x.get("series") or x.get("securityType") or "").upper() in ("SME", "ST"):
            continue
        ld = pdate(x.get("listingDate"))
        if not ld or ld > date.today():
            continue
        rows.append((ld, x.get("symbol"), price_band(x.get("issuePrice") or x.get("priceRange"))))
    rows.sort(reverse=True)
    res = []
    for ld, sym, p in rows[: n * 2]:
        g = listing_gain(sym, p)
        if g:
            res.append({"symbol": sym, **g})
        if len(res) >= n:
            break
    return res


def market_mood():
    try:
        import yfinance as yf
        c = yf.Ticker("^NSEI").history(period="3mo")["Close"]
        d5, d20 = (c.iloc[-1] / c.iloc[-6] - 1) * 100, (c.iloc[-1] / c.iloc[-21] - 1) * 100
        mood = "falling ⚠️" if d5 <= -2 or d20 <= -5 else "strong ✅" if d5 >= 1 and d20 >= 2 else "neutral"
        return {"d5": round(d5, 1), "d20": round(d20, 1), "mood": mood}
    except Exception as ex:
        print(f"market mood: {ex}"); return None


# ------------------------------------------------------------------ listing range from history
# 99 mainboard IPOs (Oct–Dec of 2023, 2024, 2025 and Sep–Nov 2026), final QIB subscription vs listing-day open.
# (min QIB, label, median %, middle-half low %, high %, share that opened above issue price %)
HISTORY = [(200, "200× or more", 62, 30, 76, 100), (100, "100–200×", 37, 9, 58, 92), (50, "50–100×", 29, 20, 36, 88),
           (25, "25–50×", 8, 1, 15, 73), (5, "5–25×", 2, -1, 11, 58), (0, "below 5×", 0, -2, 6, 48)]


def history_range(qib):
    if qib is None:
        return None
    return next(h for h in HISTORY if qib >= h[0])


# ------------------------------------------------------------------ verdict
def verdict(c, mood):
    q, h = c.get("qib"), c.get("hni")
    if q is None:
        return "No category data from NSE ❔", "check your broker app before 5 PM"
    if q < WEAK_QIB:
        return "Weak demand ❌", f"institutions below {WEAK_QIB}×"
    if q >= STRONG_QIB and (h or 0) >= STRONG_HNI:
        if mood and mood["mood"].startswith("falling"):
            return "Strong demand, weak market ⚠️", "listing could be flat if the market keeps falling"
        return "Strong demand ✅", f"institutions and HNIs both above {STRONG_QIB}×"
    return "Mixed ➖", "apply only if you'd be happy holding it"


def fx(v):
    return "–" if v is None else f"{v:,.1f}×" if v < 100 else f"{v:,.0f}×"


# ------------------------------------------------------------------ main entry (called from alerts.py)
def run(st, send, head, section, table, e, footer, pc):
    today = datetime.now(IST).date()
    log = st.setdefault("ipo_log", [])
    try:
        s = nse_session()
    except Exception as ex:
        print(f"NSE unreachable: {ex}"); return False
    changed = False

    # 1) fill in listing results for IPOs alerted earlier
    for it in log:
        if it.get("listing") is None and it.get("close_date", "") < today.isoformat():
            g = listing_gain(it["symbol"], it.get("price"), it.get("sme"))
            if g:
                it["listing"] = g; changed = True

    # 2) today's closing IPOs
    ipos = closing_today(s, today)
    if not ipos:
        print(f"No IPO closes today ({today}). Nothing to send.")
        return changed
    mood = market_mood()
    recent = recent_listings(s, 10)
    pos = sum(1 for r in recent if r["open"] > 0)

    for ipo in ipos:
        if any(x["symbol"] == ipo["symbol"] and x["close_date"] == today.isoformat() for x in log):
            print(f"{ipo['symbol']}: already sent today"); continue
        c = categories(get_json(s, f"/api/ipo-active-category?symbol={ipo['symbol']}") or {})
        if "total" not in c and ipo.get("total"):
            c["total"] = ipo["total"]
        v, why = verdict(c, mood)
        rows = [["Institutions (QIB)", fx(c.get("qib"))],
                ["HNI (all)", fx(c.get("hni"))]]
        if c.get("bhni") is not None:
            rows.append(["  HNI above ₹10L", fx(c.get("bhni"))])
        rows += [["Retail", fx(c.get("retail"))], ["Total", fx(c.get("total"))]]
        msg = head("🆕", f"IPO last day: {ipo['name']}" + (" (SME)" if ipo["sme"] else ""),
                   f"Bidding closes today · figures as of {datetime.now(IST).strftime('%I:%M %p').lstrip('0')}")
        msg += section("📊", "Subscription") + "\n" + table(rows)
        msg += "\n<i>Institutions often add a lot in the last 2 hours, so the final QIB number can be much higher.</i>"
        hr = history_range(c.get("qib"))
        if hr:
            msg += section("🔎", f"Past IPOs with QIB {hr[1]}")
            msg += f"\nTypical listing: <b>{pc(hr[2], 0)}</b> (most between {pc(hr[3], 0)} and {pc(hr[4], 0)})"
            msg += f"\n{hr[5]}% opened above issue price"
            msg += "\n<i>A range, not a forecast. Based on final QIB; today's number is still rising.</i>"
        msg += section("🌡", "Market")
        if mood:
            msg += f"\nNifty {pc(mood['d5'])} this week, {pc(mood['d20'])} this month · {e(mood['mood'])}"
        if recent:
            msg += f"\nLast {len(recent)} IPOs: {pos} opened above issue price"
            avg = sum(r["open"] for r in recent) / len(recent)
            msg += f" (average {pc(avg)})"
        done = [x for x in log if x.get("listing") and x.get("verdict", "").startswith("Strong demand ✅")]
        if len(done) >= 5:
            w = sum(1 for x in done if x["listing"]["open"] > 0)
            msg += f"\n\n📈 <b>Track record of 'Strong' so far</b>: {w} of {len(done)} opened above issue price"
        msg += f"\n\n<b>Verdict: {e(v)}</b>\n<i>{e(why)}</i>"
        msg += "\n\n⏰ Retail bids and UPI approval close at 5 PM."
        msg += "\n<i>Based on official NSE data, not GMP. Not investment advice.</i>" + footer()
        send(msg)
        log.append({"symbol": ipo["symbol"], "name": ipo["name"], "sme": ipo["sme"], "price": ipo["price"],
                    "close_date": today.isoformat(), "cat": c, "mood": (mood or {}).get("mood"),
                    "verdict": v, "listing": None})
        changed = True
    st["ipo_log"] = log[-200:]
    return changed


# ------------------------------------------------------------------ listing day (10:07 AM) and evening close
def listings_today(s, today):
    past = get_json(s, "/api/public-past-issues") or []
    if isinstance(past, dict):
        past = past.get("data") or []
    out = []
    for x in past:
        if str(x.get("series") or x.get("securityType") or "").upper() in ("SME", "ST"):
            continue
        if pdate(x.get("listingDate")) == today:
            out.append({"symbol": x.get("symbol"), "name": x.get("companyName") or x.get("symbol"),
                        "price": price_band(x.get("issuePrice") or x.get("priceRange"))})
    return out


def quote(s, symbol):
    q = get_json(s, f"/api/quote-equity?symbol={symbol}") or {}
    p = q.get("priceInfo") or {}
    o, l = num(p.get("open")), num(p.get("lastPrice"))
    return (o if o and o == o else None), (l if l and l == l else None)


def run_listing(st, send, head, section, table, e, footer, pc):
    today = datetime.now(IST).date()
    try:
        s = nse_session()
    except Exception as ex:
        print(f"NSE unreachable: {ex}"); return False
    items = listings_today(s, today)
    if not items:
        print(f"No mainboard IPO lists today ({today}). Nothing to send."); return False
    log, lst = st.setdefault("ipo_log", []), st.setdefault("ipo_listings", [])
    changed = False
    for it in items:
        if any(x["symbol"] == it["symbol"] and x["date"] == today.isoformat() for x in lst):
            print(f"{it['symbol']}: already sent"); continue
        o, last = quote(s, it["symbol"])
        if not o or not it["price"]:
            print(f"{it['symbol']}: no opening price yet"); continue
        g = (o / it["price"] - 1) * 100
        prev = next((x for x in reversed(log) if x["symbol"] == it["symbol"]), None)
        c = (prev or {}).get("cat") or categories(get_json(s, f"/api/ipo-active-category?symbol={it['symbol']}") or {})
        msg = head("🔔", f"Listing today: {it['name']}", datetime.now(IST).strftime("%a, %d %b · %I:%M %p"))
        msg += f"\n\nOpened <b>{pc(g)}</b> vs issue price"
        if last:
            msg += f"\nSince open: {pc((last / o - 1) * 100)}"
        hr = history_range(c.get("qib"))
        if hr:
            where = "✅ within range" if hr[3] <= g <= hr[4] else "⬆️ above range" if g > hr[4] else "⬇️ below range"
            msg += f"\n\nExpected from QIB {fx(c.get('qib'))}: {pc(hr[3], 0)} to {pc(hr[4], 0)} · {where}"
        if c:
            msg += f"\nSubscription: QIB {fx(c.get('qib'))} · HNI {fx(c.get('hni'))} · Retail {fx(c.get('retail'))}"
        msg += "\n\n<i>Facts only, not investment advice.</i>" + footer()
        send(msg)
        lst.append({"symbol": it["symbol"], "name": it["name"], "date": today.isoformat(), "price": it["price"],
                    "open": round(g, 1), "close": None, "qib": c.get("qib")})
        if prev is not None:
            prev["listing"] = {"date": today.isoformat(), "open": round(g, 1), "close": None}
        changed = True
    st["ipo_listings"] = lst[-200:]
    return changed


def evening_lines(st, pc, e):
    """Called by the 4:41 PM run: listing-day close for today's listings (Yahoo after the close)."""
    today = datetime.now(IST).date().isoformat()
    lines = []
    for x in st.get("ipo_listings", []):
        if x["date"] != today or x.get("close") is not None:
            continue
        g = listing_gain(x["symbol"], x["price"])
        if g and g["date"] == today:
            x["close"] = g["close"]
            for y in st.get("ipo_log", []):
                if y["symbol"] == x["symbol"] and y.get("listing"):
                    y["listing"]["close"] = g["close"]
            lines.append(f"• {e(x['name'])}: opened {pc(x['open'])}, closed <b>{pc(g['close'])}</b>")
    return lines
