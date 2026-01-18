import os
import time
import traceback

from dotenv import load_dotenv

from .telegram_watch_channels import scan_once


def main() -> None:
    """
    Runs telegram channel scan repeatedly (daemon-style).

    Persists:
    - Telethon session under current working dir (/app/data on VPS)
    - hits CSV wherever TG_HITS_CSV points (default: telegram_hits.csv)
    """
    load_dotenv()

    interval = int(os.getenv("TG_SCAN_INTERVAL_SEC", "60").strip() or "60")
    jitter = float(os.getenv("TG_SCAN_JITTER_SEC", "0").strip() or "0")
    backoff = int(os.getenv("TG_SCAN_BACKOFF_SEC", "30").strip() or "30")

    print("=== PolyMania Telegram Radar Loop ===")
    print(f"Interval: {interval}s | Backoff on error: {backoff}s | Jitter: {jitter}s")
    print("Tip: set TG_WATCH_SINCE_MINUTES=10 to scan only recent messages.")
    print()

    while True:
        try:
            hits = scan_once()
            sleep_for = interval
            if jitter > 0:
                sleep_for = max(1, int(interval + (jitter * (0.5 - time.time() % 1))))
            print(f"[TG LOOP] Scan done, hits={hits}. Sleeping {sleep_for}s...")
            time.sleep(sleep_for)
        except Exception as e:
            print("[TG LOOP] ERROR:", str(e))
            traceback.print_exc()
            print(f"[TG LOOP] Sleeping {backoff}s then retrying...")
            time.sleep(backoff)


if __name__ == "__main__":
    main()
