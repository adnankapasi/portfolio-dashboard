# Portfolio Dashboard

A password-protected daily dashboard for my holdings, hosted on GitHub Pages.

- Every weekday at 4:45 PM IST, GitHub Actions fetches prices (Yahoo Finance), calculates
  50-EMA, 200-SMA, Supertrend, RSI, ADX and checks my add/exit trigger levels.
- The results are **encrypted with AES-256** using my password and saved as `data.enc.json`.
- My holdings (quantities, buy prices) live only in the `HOLDINGS_JSON` secret, never in this repo.
- The page asks for the password and decrypts in the browser.

## Update holdings
Edit the `HOLDINGS_JSON` secret (Settings → Secrets and variables → Actions), then run the
workflow manually from the Actions tab.

## Run now
Actions → **Update dashboard** → **Run workflow**.
