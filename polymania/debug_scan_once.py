import logging
from typing import Any, Dict, List

from .config import settings
from .polymarket_client import (
    fetch_active_events,
    fetch_trades_for_event,
    current_timestamp,
)
from .detector import detect_run_for_event, normalize_trade


def _is_event_recent(event: Dict[str, Any]) -> bool:
    """
    Local copy of the recency filter logic for debugging.

    If created_at / createdAt is missing or invalid, we treat the event as recent.
    """
    created_at = event.get("created_at") or event.get("createdAt")
    if not created_at:
        return True

    try:
        created_ts = int(created_at)
    except (TypeError, ValueError):
        return True

    now = current_timestamp()
    age_sec = now - created_ts

    is_war = _is_war_conflict_event(event)
    if is_war:
        if settings.max_event_age_hours_war is None:
            return True
        return age_sec <= settings.max_event_age_hours_war * 3600

    if settings.max_event_age_hours is None:
        return True

    return age_sec <= settings.max_event_age_hours * 3600


def _is_war_conflict_event(event: Dict[str, Any]) -> bool:
    text_parts = [
        str(event.get("title", "")),
        str(event.get("slug", "")),
        str(event.get("description", "")),
    ]
    text = " ".join(text_parts).lower()

    for kw in settings.war_keywords:
        kw = kw.strip().lower()
        if kw and kw in text:
            return True
    return False


def _matches_keywords(event: Dict[str, Any]) -> bool:
    """
    Local copy of WATCH_KEYWORDS logic for debugging.

    If WATCH_KEYWORDS is empty, we watch all events.
    Otherwise, at least one keyword must appear in title or slug.
    """
    keywords = settings.watch_keywords or []
    if not keywords:
        return True

    title = (event.get("title") or "").lower()
    slug = (event.get("slug") or event.get("eventSlug") or "").lower()
    text = f"{title} {slug}"

    return any(k in text for k in keywords)


def main() -> None:
    # Configure logging to show DEBUG messages
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(levelname)s] %(name)s: %(message)s",
        force=True,
    )
    
    print("=== PolyMania Debug Scan (single pass) ===")
    print(f"POLL_INTERVAL_SEC={settings.poll_interval_sec}")
    print(f"RECENT_WINDOW_SEC={settings.recent_window_sec}")
    print(f"BASE_WINDOW_SEC={settings.base_window_sec}")
    print(f"MIN_PRICE_JUMP={settings.min_price_jump}")
    print(f"MIN_RECENT_VOLUME={settings.min_recent_volume}")
    print(f"DOMINANCE_THRESHOLD={settings.dominance_threshold}")
    print(f"MAX_EVENT_AGE_HOURS={settings.max_event_age_hours}")
    print(f"WATCH_KEYWORDS={settings.watch_keywords!r}")
    print()

    events = fetch_active_events(limit=100)
    total_events = len(events)
    print(f"Fetched {total_events} active events from Polymarket.\n")

    recent_count = 0
    keyword_count = 0
    signal_count = 0

    for event in events:
        event_id = event.get("id")
        title = event.get("title") or "(no title)"
        slug = event.get("slug") or event.get("eventSlug") or ""

        is_recent = _is_event_recent(event)
        if not is_recent:
            continue
        recent_count += 1

        if not _matches_keywords(event):
            continue
        keyword_count += 1

        print(f"--- EVENT {event_id} ---")
        print(f"Title: {title}")
        print(f"Slug:  {slug}")

        trades_raw = fetch_trades_for_event(str(event_id), limit=500)
        trades_norm = [normalize_trade(t) for t in trades_raw]
        print(f"Trades fetched (normalized): {len(trades_norm)}")

        signal = detect_run_for_event(event, trades_norm)
        if signal:
            signal_count += 1
            print(">>> SIGNAL DETECTED <<<")
            print(
                f"  outcome={signal.get('dominant_outcome')} "
                f"side={signal.get('dominant_side')} "
                f"jump={signal.get('price_jump'):.3f} "
                f"vol={signal.get('recent_volume'):.1f} "
                f"dom={signal.get('dominance'):.2f}"
            )
        else:
            print("No signal detected for this event.")

        print()

    print("=== SUMMARY ===")
    print(f"Total events fetched:      {total_events}")
    print(f"Events passing recency:    {recent_count}")
    print(f"Events passing keywords:   {keyword_count}")
    print(f"Signals detected (this run): {signal_count}")
    print()
    print("If 'Signals detected' is 0 but 'Events passing keywords' is > 0,")
    print("you may want to ease MIN_PRICE_JUMP / MIN_RECENT_VOLUME / DOMINANCE_THRESHOLD.")
    print("If 'Events passing keywords' is 0, adjust WATCH_KEYWORDS or clear it temporarily.")


if __name__ == "__main__":
    main()

