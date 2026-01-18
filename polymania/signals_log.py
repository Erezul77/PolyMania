import csv
import os
from datetime import datetime
from typing import Any, Dict, List, Optional


SIGNALS_CSV_PATH = "signals.csv"


def _ensure_header(path: str) -> None:
    """
    Ensure the CSV file exists and has a header row.
    """
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return

    header = [
        "timestamp_iso",
        "event_id",
        "event_title",
        "event_slug",
        "dominant_outcome",
        "dominant_side",
        "last_price",
        "base_price",
        "price_jump",
        "recent_volume",
        "dominance",
        "signal_type",
        "signal_reason",
        "news_titles",
    ]

    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None

    with open(path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def append_signal_row(
    ts: int,
    signal: Dict[str, Any],
    news: Optional[List[Dict[str, Any]]],
    signal_info: Optional[Dict[str, Any]],
) -> None:
    """
    Append a single signal row to signals.csv for later analysis.

    ts:      unix timestamp (seconds)
    signal:  dict produced by detect_run_for_event (plus any extra fields)
    news:    list of news dicts (can be None/empty)
    signal_info: dict from decide_signal_type (can be None)
    """
    _ensure_header(SIGNALS_CSV_PATH)

    timestamp_iso = datetime.utcfromtimestamp(ts).isoformat() + "Z"

    news_titles: List[str] = []
    if news:
        for item in news:
            title = item.get("title") or ""
            if title:
                news_titles.append(title)
    news_titles_str = " | ".join(news_titles) if news_titles else ""

    row = [
        timestamp_iso,
        signal.get("event_id"),
        signal.get("event_title"),
        signal.get("event_slug"),
        signal.get("dominant_outcome"),
        signal.get("dominant_side"),
        f"{signal.get('last_price', 0.0):.6f}",
        f"{signal.get('base_price', 0.0):.6f}",
        f"{signal.get('price_jump', 0.0):.6f}",
        f"{signal.get('recent_volume', 0.0):.6f}",
        f"{signal.get('dominance', 0.0):.6f}",
        (signal_info or {}).get("type"),
        (signal_info or {}).get("reason"),
        news_titles_str,
    ]

    with open(SIGNALS_CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

