"""Does buying mutual funds on dips beat a plain SIP? Tested on each fund's full daily NAV history.

Same total money in every strategy (Rs 10,000 per fund per month). Cash waiting for a dip earns 6%/yr
(liquid fund). Every dip strategy is run "optimistic" (buy on the dip day itself) and "realistic"
(buy the next day, because the NAV is only known in the evening).
NAV history: mfapi.in (free mirror of AMFI data).
"""
import json
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MONTHLY, LIQUID, DIP, BIG = 10000, 0.06, 0.015, 0.10
FUNDS = [  # (your name, search text, must include, must exclude)
    ("ICICI Large & Midcap", "ICICI Prudential Large & Mid Cap", ["large", "mid"], ["idcw", "dividend", "bonus"]),
    ("Nippon Multi Asset", "Nippon India Multi Asset", ["multi asset"], ["omni", "fof", "idcw", "dividend"]),
    ("Nippon Multi Asset Omni", "Nippon India Multi Asset Omni", ["omni"], ["idcw", "dividend"]),
    ("ICICI Income Plus Arbitrage Omni", "ICICI Prudential Income plus Arbitrage", ["arbitrage"], ["idcw", "dividend"]),
]


def get_json(url, **kw):
    import requests
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"}, **kw)
    r.raise_for_status()
    return r.json()


def find_scheme(search, inc, exc):
    res = get_json("https://api.mfapi.in/mf/search", params={"q": search})
    ok = [x for x in res if all(k in x["schemeName"].lower() for k in inc) and not any(k in x["schemeName"].lower() for k in exc)
          and "growth" in x["schemeName"].lower()]
    ok.sort(key=lambda x: ("direct" not in x["schemeName"].lower(), len(x["schemeName"])))
    return ok[0] if ok else None


def nav_history(code):
    d = get_json(f"https://api.mfapi.in/mf/{code}")["data"]
    s = pd.Series({pd.to_datetime(x["date"], dayfirst=True): float(x["nav"]) for x in d if float(x["nav"]) > 0}).sort_index()
    return s[~s.index.duplicated()]


# ------------------------------------------------------------------ strategies
def xirr(flows):
    """flows: list of (date, amount) with investments negative and final value positive."""
    if len(flows) < 2:
        return None
    t0 = flows[0][0]
    yrs = np.array([(d - t0).days / 365.25 for d, _ in flows]); amt = np.array([a for _, a in flows])
    lo, hi = -0.99, 3.0
    f = lambda r: (amt / (1 + r) ** yrs).sum()
    if f(lo) * f(hi) > 0:
        return None
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(lo) * f(mid) > 0 else (lo, mid)
    return mid


def simulate(nav, strategy, lag):
    """Returns dict with xirr, final value, invested, avg idle cash share."""
    months = nav.groupby(nav.index.to_period("M"))
    days = nav.index
    units, cash, flows, idle, reserve = 0.0, 0.0, [], [], 0.0
    daily_liq = (1 + LIQUID) ** (1 / 365) - 1
    last_day = None
    pos = {d: i for i, d in enumerate(days)}
    high = nav.cummax()

    def buy(day, amt):
        nonlocal units
        i = min(pos[day] + lag, len(days) - 1)
        units += amt / nav.iloc[i]

    for per, m in months:
        if last_day is not None:
            cash *= (1 + daily_liq) ** (m.index[0] - last_day).days
            reserve *= (1 + daily_liq) ** (m.index[0] - last_day).days
        flows.append((m.index[0].date(), -MONTHLY))
        start = nav.iloc[max(pos[m.index[0]] - 1, 0)]  # previous close = month's starting level
        if strategy == "sip":
            buy(m.index[min(4, len(m) - 1)], MONTHLY)  # around the 5th trading day
        elif strategy in ("flex", "flex_big"):
            amt = MONTHLY * (0.75 if strategy == "flex_big" else 1.0)
            if strategy == "flex_big":
                reserve += MONTHLY * 0.25
            dips = [d for d in m.index if nav[d] <= start * (1 - DIP)]
            deadline = m.index[max(len(m) - 3, 0)]
            buy(dips[0] if dips and dips[0] <= deadline else deadline, amt)
            if strategy == "flex_big" and reserve > 0:
                big = [d for d in m.index if nav[d] <= high[d] * (1 - BIG)]
                if big:
                    buy(big[0], reserve); reserve = 0.0
        elif strategy == "wait":
            cash += MONTHLY
            dipped = [d for d in m.index if nav[d] <= start * (1 - 0.03) or nav[d] <= high[d] * (1 - 0.05)]
            if dipped:
                buy(dipped[0], cash); cash = 0.0
        idle.append((cash + reserve) / (MONTHLY * len(flows)))
        last_day = m.index[0]  # interest is applied from one month start to the next
    grow = (1 + daily_liq) ** (nav.index[-1] - last_day).days
    value = units * nav.iloc[-1] + (cash + reserve) * grow
    flows.append((nav.index[-1].date(), value))
    inv = MONTHLY * (len(flows) - 1)
    return {"xirr": xirr(flows), "value": value, "invested": inv, "idle_pct": float(min(np.mean(idle) * 100, 100))}


STRATS = [("sip", 0, "Plain SIP (around the 5th)"), ("flex", 0, "Flexible date · optimistic"), ("flex", 1, "Flexible date · realistic"),
          ("flex_big", 0, "Flexible + big-dip extra · optimistic"), ("flex_big", 1, "Flexible + big-dip extra · realistic"),
          ("wait", 0, "Wait for dips · optimistic"), ("wait", 1, "Wait for dips · realistic")]


def windows(nav, years=5):
    """Each 5-year window starting every 6 months: how often did each strategy beat plain SIP?"""
    starts = pd.date_range(nav.index[0], nav.index[-1] - pd.DateOffset(years=years), freq="6MS")
    res = {lab: [] for _, _, lab in STRATS[1:]}
    for s0 in starts:
        part = nav[(nav.index >= s0) & (nav.index < s0 + pd.DateOffset(years=years))]
        if len(part) < 200 * years:
            continue
        base = simulate(part, "sip", 0)["xirr"]
        for st, lag, lab in STRATS[1:]:
            x = simulate(part, st, lag)["xirr"]
            if base is not None and x is not None:
                res[lab].append(x - base)
    return {k: {"n": len(v), "beat_pct": round(float(np.mean([d > 0 for d in v]) * 100)), "avg_gap_pct": round(float(np.mean(v) * 100), 2)}
            for k, v in res.items() if v}


def dip_months(nav):
    """Share of months where the NAV fell at least DIP below the previous month's last NAV."""
    prev, hits, n = None, 0, 0
    for _, m in nav.groupby(nav.index.to_period("M")):
        if prev is not None:
            n += 1; hits += bool((m <= prev * (1 - DIP)).any())
        prev = m.iloc[-1]
    return round(hits / n * 100) if n else None


def main():
    out, md = {"generated": date.today().isoformat(), "monthly": MONTHLY, "liquid_rate": LIQUID, "funds": []}, []
    md.append(f"# Dip buying vs SIP · ₹{MONTHLY:,} a month per fund · idle cash earns {LIQUID * 100:.0f}%\n")
    for name, q, inc, exc in FUNDS:
        try:
            sch = find_scheme(q, inc, exc)
            if not sch:
                md.append(f"## {name}\nCouldn't find this fund on mfapi.in.\n"); continue
            nav = nav_history(sch["schemeCode"])
        except Exception as e:
            md.append(f"## {name}\nDownload failed: {e}\n"); continue
        yrs = (nav.index[-1] - nav.index[0]).days / 365.25
        f = {"name": name, "scheme": sch["schemeName"], "code": sch["schemeCode"], "from": str(nav.index[0].date()), "years": round(yrs, 1),
             "daily_vol_pct": round(float(nav.pct_change().std() * 100), 2),
             "pct_months_with_dip": dip_months(nav), "results": []}
        base = None
        for st, lag, lab in STRATS:
            r = simulate(nav, st, lag)
            base = base or r
            r.update(label=lab, gap_pct=None if r["xirr"] is None or base["xirr"] is None else round((r["xirr"] - base["xirr"]) * 100, 2))
            f["results"].append(r)
        f["windows_5y"] = windows(nav) if yrs >= 7 else {}
        out["funds"].append(f)
        md.append(f"## {name}\n_{sch['schemeName']} · NAV since {f['from']} ({f['years']} yrs) · typical daily move {f['daily_vol_pct']}% · "
                  f"{f['pct_months_with_dip']}% of months had a {DIP * 100:.1f}% dip_\n")
        md.append("| Strategy | XIRR | vs SIP | Final value | Idle cash |\n|---|---|---|---|---|")
        for i, r in enumerate(f["results"]):
            if r["xirr"] is None:
                md.append(f"| {r['label']} | – | | | |"); continue
            gap = "" if i == 0 or r["gap_pct"] is None else f"{r['gap_pct']:+.2f}%"
            md.append(f"| {r['label']} | {r['xirr'] * 100:.2f}% | {gap} | ₹{r['value']:,.0f} of ₹{r['invested']:,.0f} | {r['idle_pct']:.0f}% |")
        if f["windows_5y"]:
            md.append("\n**Every 5-year period (starting every 6 months):** how often each beat plain SIP\n\n| Strategy | Periods | Beat SIP | Avg XIRR gap |\n|---|---|---|---|")
            for k, v in f["windows_5y"].items():
                md.append(f"| {k} | {v['n']} | {v['beat_pct']}% | {v['avg_gap_pct']:+.2f}% |")
        md.append("")
    md.append("_Optimistic = buy on the dip day itself. Realistic = buy the next day (NAV is only known in the evening). Not financial advice._")
    (ROOT / "mf_test.json").write_text(json.dumps(out, default=float, indent=1))
    text = "\n".join(md)
    (ROOT / "mf_test_summary.md").write_text(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        open(os.environ["GITHUB_STEP_SUMMARY"], "a").write(text)
    print(text)


if __name__ == "__main__":
    main()
