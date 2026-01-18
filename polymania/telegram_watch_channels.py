import os
import csv
import time
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Optional

from dotenv import load_dotenv
from telethon.sync import TelegramClient
from telethon.errors.rpcerrorlist import FloodWaitError

from .notifier import notify_telegram_hit


def _parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in (raw or "").split(",") if x.strip()]


def _require_env(name: str) -> str:
    val = os.getenv(name, "").strip()
    if not val:
        raise RuntimeError(f"Missing {name} in .env")
    return val


def _normalize_channel_ref(ref: str) -> str:
    """
    Accepts:
    - @channelusername
    - channelusername
    - https://t.me/channelusername
    - t.me/channelusername
    - numeric ids (as string)
    Returns something Telethon get_entity can resolve.
    """
    r = ref.strip()
    if not r:
        return r

    # Strip URL parts
    for prefix in ("https://t.me/", "http://t.me/", "https://telegram.me/", "http://telegram.me/"):
        if r.startswith(prefix):
            r = r[len(prefix):]

    if r.startswith("t.me/"):
        r = r[len("t.me/"):]

    # Remove leading @ for uniformity (Telethon accepts both)
    if r.startswith("@"):
        r = r[1:]

    return r


def _get_scan_limit() -> int:
    # Support a few env names for compatibility
    for key in ("TG_WATCH_LIMIT", "TG_SCAN_LIMIT", "TG_LIMIT"):
        v = os.getenv(key, "").strip()
        if v:
            try:
                return int(v)
            except ValueError:
                pass
    return 200


def _get_since_dt() -> Optional[datetime]:
    """
    Optional time filter:
      - TG_WATCH_SINCE_MINUTES
      - TG_WATCH_SINCE_HOURS
    If set, we stop scanning older messages beyond that time.
    """
    mins = os.getenv("TG_WATCH_SINCE_MINUTES", "").strip()
    hrs = os.getenv("TG_WATCH_SINCE_HOURS", "").strip()

    now = datetime.now(timezone.utc)

    if mins:
        try:
            m = int(mins)
            return now - timedelta(minutes=m)
        except ValueError:
            return None

    if hrs:
        try:
            h = int(hrs)
            return now - timedelta(hours=h)
        except ValueError:
            return None

    return None


def _ensure_csv_header(path: str) -> None:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return

    header = [
        "scan_timestamp_utc",
        "channel_title",
        "channel_username",
        "channel_id",
        "message_id",
        "message_date_utc",
        "matched_keyword",
        "text_excerpt",
        "message_url",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)


def _excerpt(text: str, n: int = 180) -> str:
    t = (text or "").replace("\n", " ").replace("\r", " ").strip()
    if len(t) <= n:
        return t
    return t[: n - 1] + "…"


def _build_msg_url(username: Optional[str], msg_id: int) -> str:
    if username:
        return f"https://t.me/{username}/{msg_id}"
    return ""


def scan_once() -> int:
    load_dotenv()

    api_id = int(_require_env("TG_API_ID"))
    api_hash = _require_env("TG_API_HASH")
    session_name = os.getenv("TG_SESSION_NAME", "polymania_user").strip() or "polymania_user"

    raw_channels = _require_env("TG_WATCH_CHANNELS")
    raw_keywords = _require_env("TG_WATCH_KEYWORDS")

    channels = [_normalize_channel_ref(x) for x in _parse_csv_list(raw_channels)]
    keywords = _parse_csv_list(raw_keywords)

    # Casefold for robust matching across languages (Arabic unaffected, English improves)
    keywords_cf = [(k, k.casefold()) for k in keywords]

    limit = _get_scan_limit()
    hits_csv = os.getenv("TG_HITS_CSV", "telegram_hits.csv").strip() or "telegram_hits.csv"
    sleep_sec = float(os.getenv("TG_WATCH_SLEEP_SEC", "1").strip() or "1")

    since_dt = _get_since_dt()

    print("=== PolyMania Telegram Scan (channels once) ===")
    print(f"Session: {session_name!r}")
    print(f"Channels: {len(channels)} | Keywords: {len(keywords)} | Limit/channel: {limit}")
    if since_dt:
        print(f"Since: {since_dt.isoformat()}")
    print(f"Output CSV: {hits_csv}")
    print()

    _ensure_csv_header(hits_csv)

    total_hits = 0
    scan_ts = datetime.now(timezone.utc).isoformat()

    with TelegramClient(session_name, api_id, api_hash) as client:
        me = client.get_me()
        print(f"Logged in as: {getattr(me, 'first_name', 'Unknown')} (@{getattr(me, 'username', '')}) id={me.id}")
        print()

        for ref in channels:
            try:
                entity = client.get_entity(ref)
            except Exception as e:
                print(f"[WARN] Cannot resolve channel {ref!r}: {e}")
                continue

            title = getattr(entity, "title", None) or str(ref)
            username = getattr(entity, "username", None)
            channel_id = getattr(entity, "id", None)

            print(f"\n=== Scanning channel: {title!r} (id={channel_id}) ===")

            try:
                for msg in client.iter_messages(entity, limit=limit):
                    if not msg:
                        continue

                    # Optional time cutoff (messages are newest->oldest)
                    if since_dt and msg.date and msg.date.replace(tzinfo=timezone.utc) < since_dt:
                        break

                    text = msg.message or ""
                    if not text:
                        continue

                    text_cf = text.casefold()

                    matched: List[Tuple[str, str]] = []
                    for orig, kcf in keywords_cf:
                        if kcf and kcf in text_cf:
                            matched.append((orig, kcf))

                    if not matched:
                        continue

                    # Write one row per matched keyword (keeps analysis simple)
                    msg_date = msg.date.replace(tzinfo=timezone.utc).isoformat() if msg.date else ""
                    url = _build_msg_url(username, msg.id)

                    for orig, _ in matched:
                        total_hits += 1
                        row = [
                            scan_ts,
                            title,
                            username or "",
                            channel_id or "",
                            msg.id,
                            msg_date,
                            orig,
                            _excerpt(text),
                            url,
                        ]
                        with open(hits_csv, "a", newline="", encoding="utf-8") as f:
                            w = csv.writer(f)
                            w.writerow(row)

                        print(f"[HIT] {orig!r} | {msg_date} | {url}")
                        print(f"      {_excerpt(text, 140)}")
                        
                        # Send Telegram alert for this hit
                        try:
                            notify_telegram_hit(
                                keyword=orig,
                                channel_title=title,
                                channel_username=username,
                                message_text=text,
                                message_url=url,
                                message_date=msg_date,
                            )
                        except Exception as e:
                            print(f"[WARN] Failed to send Telegram alert: {e}")

            except FloodWaitError as e:
                wait_s = int(getattr(e, "seconds", 10))
                print(f"[WARN] FloodWait: sleeping {wait_s}s then continuing...")
                time.sleep(wait_s)
            except Exception as e:
                print(f"[WARN] Error scanning {title!r}: {e}")

            time.sleep(sleep_sec)

    print(f"\nDone. Total hits: {total_hits}")
    return total_hits


def main() -> None:
    try:
        scan_once()
    except Exception as e:
        print("ERROR:", e)
        raise


if __name__ == "__main__":
    main()
