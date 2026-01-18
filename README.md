# PolyMania

Local tool that watches Polymarket events, detects sharp "runs" in price/volume,
and sends you alerts (Telegram or console) together with recent news headlines
about the event.

## Quickstart

1. Create and activate a virtual environment:

```bash
python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and fill in:

   - `TELEGRAM_BOT_TOKEN` – bot token from @BotFather (optional)
   - `TELEGRAM_CHAT_ID` – your chat/user ID (optional)
   - `NEWS_API_KEY` – API key from https://newsapi.org (optional)

   Without these, PolyMania will still run and print alerts to the console.

4. Run:

```bash
python -m polymania.main
```

## Notes

- This tool is for research/educational purposes only.
- Check local laws and Polymarket's Terms of Service before automating trading or making financial decisions based on this data.

---

## Running with Cursor Tasks

If you are using Cursor or VSCode, you can use the built-in tasks to manage
PolyMania:

1. Open the command palette and choose **Tasks: Run Task**.
2. Run **PolyMania: Create venv** once to create the `.venv` environment.
3. Run **PolyMania: Install requirements** to install dependencies.
4. Run **PolyMania: Run (using .venv)** to start the main loop.

Alternatively, you can use the debugger:

1. Make sure the Python interpreter is set to the `.venv` environment
   (bottom-right status bar in Cursor/VSCode).
2. Choose the **PolyMania: Run main** launch configuration.
3. Press **F5** to launch PolyMania with the debugger.

---

## Filters & cooldown (v0.2)

PolyMania now supports a few extra controls via the `.env` file:

- `WATCH_KEYWORDS` – comma-separated list of keywords.  
  Only events whose title or slug contain at least one of these keywords
  will be monitored. If left empty, all active events are scanned.

- `COOLDOWN_SEC` – minimum number of seconds between alerts for the same
  event. This prevents spam when a market keeps spiking. The default is
  `300` seconds (5 minutes).

- `LOG_FILE` – path to a log file. By default it is `logs/polymania.log`.
  Logs are written both to the console and to this file using a rotating
  log handler.

Adjust these values in your `.env` to tune how sensitive and noisy
PolyMania should be for your use case.

