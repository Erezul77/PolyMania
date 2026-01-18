from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

from .config import settings


TOPIC_GROUPS: Dict[str, List[str]] = {
    # War / Middle East cluster (what the user explicitly cares about)
    "MIDEAST_WAR": [
        "gaza",
        "israel",
        "hezbollah",
        "lebanon",
        "idf",
        "hamas",
        "middle east",
        "iran",
    ],
    # Elections / politics
    "ELECTIONS": [
        "election",
        "elections",
        "president",
        "presidency",
        "primary",
        "primaries",
        "runoff",
        "trump",
        "biden",
        "vote",
        "ballot",
    ],
    # Crypto / markets
    "CRYPTO": [
        "crypto",
        "bitcoin",
        "btc",
        "ethereum",
        "eth",
        "solana",
        "sol",
        "xrp",
        "etf",
    ],
    # AI / tech / regulation
    "AI_TECH": [
        "ai",
        "artificial intelligence",
        "openai",
        "xai",
        "anthropic",
        "google",
        "meta",
        "llm",
        "gpt",
        "deepseek",
    ],
}


def detect_topic_for_event(title: str, slug: str) -> Optional[str]:
    """
    Detect a rough 'topic group' for a given event based on title and slug.
    """
    text = f"{title} {slug}".lower()
    for group, keywords in TOPIC_GROUPS.items():
        for kw in keywords:
            if kw in text:
                return group
    return None


@dataclass
class StoredSignal:
    ts: int
    topic: str
    event_id: str
    event_title: str
    direction: str  # "UP" / "DOWN" / ""
    price_jump: float
    abs_price_jump: float
    dominance: float
    recent_volume: float


class CorrelationTracker:
    """
    Tracks recent signals and forms 'clusters' when multiple signals
    in the same topic appear within a given time window.
    """

    def __init__(self, window_sec: int, min_signals_per_topic: int, cooldown_sec: int) -> None:
        self.window_sec = window_sec
        self.min_signals_per_topic = min_signals_per_topic
        self.cooldown_sec = cooldown_sec

        self._signals: Deque[StoredSignal] = deque()
        self._last_emitted_ts: Dict[str, int] = {}

    def add_signal(self, ts: int, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Register a new run signal and, if it forms part of a 'cluster' in the same
        topic, return a correlation meta-signal dict. Otherwise return None.
        """
        title = str(signal.get("event_title") or "")
        slug = str(signal.get("event_slug") or "")
        topic = detect_topic_for_event(title, slug)
        if not topic:
            return None

        # Prune old signals outside the window
        cutoff = ts - self.window_sec
        self._signals = deque(s for s in self._signals if s.ts >= cutoff)

        direction = str(signal.get("move_direction") or "").upper()
        price_jump = float(signal.get("price_jump") or 0.0)
        abs_price_jump = float(signal.get("abs_price_jump") or abs(price_jump))
        dominance = float(signal.get("dominance") or 0.0)
        recent_volume = float(signal.get("recent_volume") or 0.0)
        event_id = str(signal.get("event_id") or "")

        stored = StoredSignal(
            ts=ts,
            topic=topic,
            event_id=event_id,
            event_title=title,
            direction=direction,
            price_jump=price_jump,
            abs_price_jump=abs_price_jump,
            dominance=dominance,
            recent_volume=recent_volume,
        )
        self._signals.append(stored)

        # Collect cluster for this topic
        cluster_signals: List[StoredSignal] = [s for s in self._signals if s.topic == topic]
        if len(cluster_signals) < self.min_signals_per_topic:
            return None

        # Cooldown per topic
        last_ts = self._last_emitted_ts.get(topic, 0)
        if ts - last_ts < self.cooldown_sec:
            return None

        self._last_emitted_ts[topic] = ts

        up_count = sum(1 for s in cluster_signals if s.direction == "UP")
        down_count = sum(1 for s in cluster_signals if s.direction == "DOWN")
        avg_abs_jump = sum(s.abs_price_jump for s in cluster_signals) / len(cluster_signals)
        avg_dominance = sum(s.dominance for s in cluster_signals) / len(cluster_signals)

        if up_count > down_count:
            cluster_direction = "MOSTLY_UP"
        elif down_count > up_count:
            cluster_direction = "MOSTLY_DOWN"
        else:
            cluster_direction = "MIXED"

        # Build a compact event list
        event_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for s in cluster_signals:
            key = (s.event_id, s.event_title)
            if key not in event_map:
                event_map[key] = {
                    "event_id": s.event_id,
                    "event_title": s.event_title,
                }

        meta = {
            "topic": topic,
            "count": len(cluster_signals),
            "up_count": up_count,
            "down_count": down_count,
            "cluster_direction": cluster_direction,
            "avg_abs_jump": avg_abs_jump,
            "avg_dominance": avg_dominance,
            "window_sec": self.window_sec,
            "events": list(event_map.values()),
        }
        return meta


def build_default_correlation_tracker() -> Optional[CorrelationTracker]:
    """
    Helper to construct a tracker from settings, or disable correlation if
    corr_enabled is False.
    """
    if not getattr(settings, "corr_enabled", True):
        return None

    return CorrelationTracker(
        window_sec=settings.corr_window_sec,
        min_signals_per_topic=settings.corr_min_signals_per_topic,
        cooldown_sec=settings.corr_cooldown_sec,
    )
