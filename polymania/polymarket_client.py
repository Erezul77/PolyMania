import time
from typing import Any, Dict, List

import requests

from .config import settings


def fetch_active_events(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Fetch active (open) events from Polymarket Gamma API.
    
    Docs: https://docs.polymarket.com/developers/gamma-markets-api/get-events
    """
    params = {
        "order": "id",
        "ascending": "false",
        "closed": "false",
        "limit": limit,
    }
    resp = requests.get(f"{settings.gamma_base}/events", params=params, timeout=10)
    resp.raise_for_status()
    events = resp.json()
    # Some gamma endpoints wrap in {"events":[...]} - handle both
    if isinstance(events, dict) and "events" in events:
        return events["events"]
    return events


def fetch_trades_for_event(event_id: str, limit: int = 500) -> List[Dict[str, Any]]:
    """
    Fetch recent trades for a specific event from Polymarket Data API.
    
    Docs: https://docs.polymarket.com/api-reference/core/get-trades-for-a-user-or-markets
    Uses /trades with eventId filter.
    """
    params = {
        "limit": limit,
        "eventId": str(event_id),
        "takerOnly": "true",
    }
    resp = requests.get(f"{settings.data_base}/trades", params=params, timeout=10)
    resp.raise_for_status()
    trades = resp.json()
    if not isinstance(trades, list):
        return []
    return trades


def current_timestamp() -> int:
    return int(time.time())

