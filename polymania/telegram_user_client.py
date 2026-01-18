import os

from telethon import TelegramClient


def get_telegram_user_client() -> TelegramClient:
    """
    Create a Telethon TelegramClient using user API (api_id/api_hash).

    This is a USER client, not a Bot. It will create a local .session file
    (named by TG_SESSION_NAME) in the working directory, and reuse it on
    subsequent runs.
    """
    api_id_str = os.getenv("TG_API_ID") or ""
    api_hash = os.getenv("TG_API_HASH") or ""
    session_name = os.getenv("TG_SESSION_NAME", "polymania_session")

    if not api_id_str or not api_hash:
        raise RuntimeError(
            "TG_API_ID or TG_API_HASH not set in environment (.env). "
            "Please fill them with the values from my.telegram.org."
        )

    try:
        api_id = int(api_id_str)
    except ValueError:
        raise RuntimeError("TG_API_ID must be an integer")

    client = TelegramClient(session_name, api_id, api_hash)
    return client


# Backwards-compatible alias used by other helpers
def get_telegram_client() -> TelegramClient:
    return get_telegram_user_client()
