import os
import time
import logging
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional

from .config import Settings, settings
from .correlation import build_default_correlation_tracker
from .correlation_log import append_correlation_row
from .detector import detect_run_for_event, normalize_trade
from .news_client import search_news
from .notifier import notify_run, notify_correlation_cluster
from .polymarket_client import (
    fetch_active_events,
    fetch_trades_for_event,
    current_timestamp,
)
from .actions import decide_signal_type
from .signals_log import append_signal_row


def _setup_logging() -> logging.Logger:
    """
    Configure a logger that writes both to console and to a rotating log file.
    """
    logger = logging.getLogger("polymania")

    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    level_name = (settings.log_level or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (optional, can be disabled by setting LOG_FILE empty)
    log_file = settings.log_file
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        fh = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


logger = _setup_logging()

# Per-event cooldown tracking: event_id -> last alert timestamp
last_signal_ts_by_event: Dict[Any, int] = {}
correlation_tracker = build_default_correlation_tracker()


def _is_war_conflict_event(event: dict, settings: Settings) -> bool:
    """
    Return True if this event looks like a long-running war/conflict market
    (Gaza, Lebanon, Hezbollah, Hamas, Israel, strikes, ceasefire, etc),
    based on WAR_KEYWORDS from config.
    """
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


def _is_event_recent(event: Dict[str, Any]) -> bool:
    """
    Filter events to only those created in the last N hours, with a relaxed
    age policy for war/conflict markets.

    Gamma events usually include a created_at timestamp in seconds.
    If it's missing, we don't filter by age.
    """
    created_at = event.get("created_at") or event.get("createdAt")
    if not created_at:
        return True

    try:
        created_ts = int(created_at)
    except (TypeError, ValueError):
        return True

    now_ts = current_timestamp()
    age_sec = now_ts - created_ts

    is_war = _is_war_conflict_event(event, settings)
    normal_limit_hours = settings.max_event_age_hours
    war_limit_hours = settings.max_event_age_hours_war

    if is_war:
        if war_limit_hours is None:
            if normal_limit_hours is not None and age_sec > normal_limit_hours * 3600:
                logger.debug(
                    "WAR_EVENT override: keeping event_id=%s title=%r age_hours=%.2f",
                    event.get("id"),
                    event.get("title"),
                    age_sec / 3600.0,
                )
            return True

        allowed = age_sec <= war_limit_hours * 3600
        if (
            allowed
            and normal_limit_hours is not None
            and age_sec > normal_limit_hours * 3600
        ):
            logger.debug(
                "WAR_EVENT override: keeping event_id=%s title=%r age_hours=%.2f",
                event.get("id"),
                event.get("title"),
                age_sec / 3600.0,
            )
        return allowed

    if normal_limit_hours is None:
        return True

    return age_sec <= normal_limit_hours * 3600


def _should_watch_event(event: Dict[str, Any]) -> bool:
    """
    Check whether this event matches the WATCH_KEYWORDS filter.

    If WATCH_KEYWORDS is empty, we watch all events.
    Otherwise we require at least one keyword to appear in the title or slug.
    """
    keywords = settings.watch_keywords or []
    if not keywords:
        return True

    title = (event.get("title") or "").lower()
    slug = (event.get("slug") or event.get("eventSlug") or "").lower()
    text = f"{title} {slug}"

    return any(k in text for k in keywords)


def main_loop() -> None:
    logger.info(
        "PolyMania starting: poll_interval=%ss, recent_window=%ss, cooldown=%ss",
        settings.poll_interval_sec,
        settings.recent_window_sec,
        settings.cooldown_sec,
    )

    while True:
        try:
            events = fetch_active_events(limit=100)
            now = current_timestamp()
            logger.debug("Fetched %d active events from Polymarket", len(events))

            for event in events:
                if not _is_event_recent(event):
                    continue
                if not _should_watch_event(event):
                    continue

                event_id = event.get("id")
                if not event_id:
                    continue

                trades_raw = fetch_trades_for_event(str(event_id), limit=500)
                trades_norm = [normalize_trade(t) for t in trades_raw]

                signal = detect_run_for_event(event, trades_norm)
                if not signal:
                    continue

                event_key = signal.get("event_id")
                last_ts = last_signal_ts_by_event.get(event_key, 0)
                if now - last_ts < settings.cooldown_sec:
                    logger.debug(
                        "Skipping event %s due to cooldown (last=%s, now=%s)",
                        event_key,
                        last_ts,
                        now,
                    )
                    continue

                # Update cooldown timestamp
                last_signal_ts_by_event[event_key] = now

                # Build query for external news using event title / slug
                query_parts: List[str] = []
                title = signal.get("event_title")
                slug = signal.get("event_slug")
                if title:
                    query_parts.append(str(title))
                if slug and slug not in (title or ""):
                    query_parts.append(str(slug))

                news_query = " ".join(query_parts) if query_parts else "Polymarket"
                news = search_news(news_query, max_results=3)

                # Dry-run classification of the signal
                signal_info = decide_signal_type(signal, news)
                signal_type = signal_info.get("type")
                signal_reason = signal_info.get("reason")

                # Attach to signal so notifier can show it
                signal["signal_type"] = signal_type
                signal["signal_reason"] = signal_reason

                # Correlation / cluster detection (meta-signal layer)
                if correlation_tracker is not None:
                    cluster = correlation_tracker.add_signal(now, signal)
                    if cluster:
                        logger.info(
                            "Correlation cluster detected: topic=%s, count=%s, direction=%s",
                            cluster.get("topic"),
                            cluster.get("count"),
                            cluster.get("cluster_direction"),
                        )
                        append_correlation_row(now, cluster)
                        notify_correlation_cluster(cluster)

                logger.info(
                    "Run detected on event '%s' (id=%s), outcome=%s/%s, price_jump=%.3f, volume=%.2f, signal_type=%s",
                    signal.get("event_title"),
                    event_key,
                    signal.get("dominant_outcome"),
                    signal.get("dominant_side"),
                    signal.get("price_jump"),
                    signal.get("recent_volume"),
                    signal_type,
                )

                # Persist to CSV log for later analysis
                append_signal_row(now, signal, news, signal_info)

                # Notify via Telegram / console
                notify_run(signal, news)

        except Exception as e:
            logger.exception("Error in main loop: %s", e)

        time.sleep(settings.poll_interval_sec)


if __name__ == "__main__":
    main_loop()
