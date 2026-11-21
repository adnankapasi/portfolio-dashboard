# Portfolio Dashboard

A password-protected dashboard for my Kite and Groww holdings, hosted on GitHub Pages.

**Tabs:** Home (totals, value history, what needs attention) · Holdings (sort, filter, search) ·
Signals (add / hold / wait / don't add / exit) · Health (sector, market cap, concentration) ·
Market (indices, VIX, breadth, top movers, sectors, gold and silver). Tap any stock for its chart,
levels and fundamentals.

## How it works
- Every weekday at 4:45 PM IST, GitHub Actions runs `scripts/update.py`:
  prices from the NSE official bhavcopy (BSE, then Yahoo as backup), 1 year of history from Yahoo,
  indicators, signals, fundamentals (weekly) and market data.
- Everything personal is encrypted with AES-256 using my password:
  `holdings.enc.json` (my holdings), `data.enc.json` (the dashboard), `state.enc.json` (history).
- The page decrypts in the browser. Without the password, the files are unreadable.

## Update my holdings (after buying or selling)
1. Download the holdings file from Kite (Holdings → ⋯ → Download) or Groww (Holdings → Download).
2. On the dashboard, tap **Import**, pick the account, choose the file, check the preview, tap **Save**.
3. The dashboard refreshes itself in about 2 minutes.

The first import asks for a GitHub token (one time per device):
GitHub → Settings → Developer settings → Fine-grained tokens → Generate new token →
Repository access: *Only select repositories* → `portfolio-dashboard` →
Permissions: **Contents: Read and write** → Generate. Paste it into the Import screen.

## Change hand-set levels
Hand-set add/exit levels (for example HUDCO 165 / 152.5) are kept when you re-import.
Holdings without hand-set levels get automatic levels from the Supertrend and 50-EMA, updated daily.

## Run now
Actions → **Update dashboard** → **Run workflow**.
