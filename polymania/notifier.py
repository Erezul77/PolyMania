from typing import Any, Dict, List, Optional

import requests

from .config import settings


def _format_signal_message(signal: Dict[str, Any], news: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    Build a human-readable alert message for Telegram / console.
    """
    # Derive direction and percent move from signal
    price_jump = signal.get("price_jump")
    abs_price_jump = signal.get("abs_price_jump")
    base_price = signal.get("base_price")
    last_price = signal.get("last_price")
    
    # Determine direction and emoji
    if price_jump is not None:
        move_direction = signal.get("move_direction")
        if not move_direction:
            move_direction = "UP" if price_jump >= 0 else "DOWN"
        signed_pct = price_jump * 100.0
        abs_pct = abs(price_jump) * 100.0
    else:
        move_direction = signal.get("move_direction") or "UP"
        signed_pct = None
        abs_pct = None
    
    direction_emoji = "⬆️" if move_direction == "UP" else "⬇️"
    
    lines: List[str] = []
    lines.append(f"{direction_emoji} *PolyMania – {move_direction} run detected on Polymarket* {direction_emoji}")
    lines.append("")
    lines.append(f"*Event:* {signal.get('event_title')} (`{signal.get('event_id')}`)")
    if signal.get("event_slug"):
        lines.append(f"*Market:* https://polymarket.com/event/{signal['event_slug']}")
    lines.append("")
    
    # Add direction summary with signed percentage and price movement
    if signed_pct is not None and base_price is not None and last_price is not None:
        lines.append(
            f"*Direction:* {move_direction} ({signed_pct:+.2f}%) "
            f"[{base_price:.3f} → {last_price:.3f}]"
        )
    elif signed_pct is not None:
        lines.append(f"*Direction:* {move_direction} ({signed_pct:+.2f}%)")
    
    lines.append(f"*Outcome:* {signal['dominant_outcome']} ({signal['dominant_side']})")
    
    recent_volume = signal.get("recent_volume")
    dominance = signal.get("dominance")
    if recent_volume is not None and dominance is not None:
        lines.append(f"*Volume:* {recent_volume:.1f} · *Dominance:* {dominance*100:.1f}%")

    # Optional dry-run signal classification
    signal_type = signal.get("signal_type")
    signal_reason = signal.get("signal_reason")
    if signal_type:
        lines.append("")
        lines.append("*Signal type (dry-run, not trading advice):*")
        lines.append(f"`{signal_type}`")
        if signal_reason:
            lines.append(f"_Reason:_ {signal_reason}")

    if news:
        lines.append("")
        lines.append("*Latest news:*")
        for i, item in enumerate(news, start=1):
            title = item.get("title") or "Untitled"
            src = item.get("source") or "Unknown source"
            url = item.get("url") or ""
            lines.append(f"{i}. {title} — _{src}_")
            if url:
                lines.append(url)

    return "\n".join(lines)


def send_telegram_message(text: str) -> bool:
    """
    Send a Markdown-formatted message to a Telegram chat.

    Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in env.
    """
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        # No config – treat as no-op
        print("[PolyMania] Telegram not configured, printing message instead:\n")
        print(text)
        print("\n" + "=" * 80 + "\n")
        return False

    url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
    payload = {
        "chat_id": settings.telegram_chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }

    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        print("[PolyMania] Failed to send Telegram message:", e)
        print("Message was:\n", text)
        return False


def notify_run(signal: Dict[str, Any], news: Optional[List[Dict[str, Any]]] = None) -> None:
    """
    High-level notification helper.
    """
    msg = _format_signal_message(signal, news)
    send_telegram_message(msg)


def _format_correlation_message(cluster: Dict[str, Any]) -> str:
    """
    Build a human-readable alert message for correlation clusters.
    """
    topic = str(cluster.get("topic") or "")
    count = int(cluster.get("count") or 0)
    direction = str(cluster.get("cluster_direction") or "MIXED")
    avg_abs_jump = float(cluster.get("avg_abs_jump") or 0.0)
    avg_dom = float(cluster.get("avg_dominance") or 0.0) * 100.0
    window_sec = int(cluster.get("window_sec") or 0)
    events = cluster.get("events") or []

    if direction == "MOSTLY_UP":
        arrow = "⬆️"
    elif direction == "MOSTLY_DOWN":
        arrow = "⬇️"
    else:
        arrow = "🔀"

    lines: List[str] = []
    lines.append(f"{arrow} *PolyMania – Correlated move detected* {arrow}")
    lines.append("")
    lines.append(f"*Topic:* `{topic}`")
    lines.append(f"*Signals in window:* {count} over last {window_sec}s")
    lines.append(f"*Cluster direction:* `{direction}`")
    lines.append(f"*Avg move size:* {avg_abs_jump*100:.2f}%")
    lines.append(f"*Avg dominance:* {avg_dom:.1f}%")

    if events:
        lines.append("")
        lines.append("*Markets involved:*")
        for e in events:
            title = e.get("event_title") or "Untitled"
            eid = e.get("event_id") or "?"
            lines.append(f"- {title} (`{eid}`)")

    return "\n".join(lines)


def notify_correlation_cluster(cluster: Dict[str, Any]) -> None:
    """
    High-level notification helper for correlation clusters.
    """
    msg = _format_correlation_message(cluster)
    send_telegram_message(msg)


def notify_trading_signal(signal_message: str) -> None:
    """
    Send trading signal alert to Telegram.
    
    Args:
        signal_message: Pre-formatted signal message (from trading_signals.format_signal_for_telegram)
    """
    send_telegram_message(signal_message)


def notify_telegram_hit(
    keyword: str,
    channel_title: str,
    channel_username: Optional[str],
    message_text: str,
    message_url: str,
    message_date: str,
) -> None:
    """
    Send alert for Telegram channel keyword match.
    
    Args:
        keyword: The matched keyword
        channel_title: Channel name/title
        channel_username: Channel username (if available)
        message_text: Excerpt of the message
        message_url: Link to the message
        message_date: ISO timestamp of the message
    """
    # Format the alert message
    lines: List[str] = []
    lines.append("📡 *PolyMania – Telegram Radar Hit* 📡")
    lines.append("")
    lines.append(f"*Keyword:* `{keyword}`")
    lines.append(f"*Channel:* {channel_title}")
    if channel_username:
        lines.append(f"*Username:* @{channel_username}")
    lines.append(f"*Time:* {message_date}")
    lines.append("")
    lines.append("*Message excerpt:*")
    
    # Truncate message if too long (Telegram has limits)
    max_len = 300
    if len(message_text) > max_len:
        excerpt = message_text[:max_len] + "..."
    else:
        excerpt = message_text
    
    lines.append(f"_{excerpt}_")
    lines.append("")
    lines.append(f"[View message]({message_url})")
    
    msg = "\n".join(lines)
    send_telegram_message(msg)
