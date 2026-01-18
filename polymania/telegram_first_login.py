import os
import sys

from dotenv import load_dotenv
from telethon.sync import TelegramClient

load_dotenv()


def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        print(f"ERROR: Missing {name} in .env")
        sys.exit(1)
    return val


def main() -> None:
    """
    One-time interactive login for Telethon user session.

    Run from project root with venv active:
        python -m polymania.telegram_first_login

    It will prompt for:
    - phone number (+972...)
    - login code
    - (optional) 2FA password

    Then it creates a .session file based on TG_SESSION_NAME.
    """
    api_id_str = _require_env("TG_API_ID")
    api_hash = _require_env("TG_API_HASH")
    session_name = os.getenv("TG_SESSION_NAME", "polymania_user")

    try:
        api_id = int(api_id_str)
    except ValueError:
        print("ERROR: TG_API_ID must be a number")
        sys.exit(1)

    print("=== PolyMania Telegram FIRST LOGIN ===")
    print(f"Session name: {session_name!r}  (will create {session_name}.session)")
    print("If asked, enter your phone in international format, e.g. +972XXXXXXXXX")
    print()

    with TelegramClient(session_name, api_id, api_hash) as client:
        me = client.get_me()
        username = getattr(me, "username", None)
        name = getattr(me, "first_name", None) or "Unknown"
        print(f"✅ Logged in as: {name} @{username} (id={me.id})" if username else f"✅ Logged in as: {name} (id={me.id})")

    print("\nDone. Session saved. Do NOT commit .session files to git.")


if __name__ == "__main__":
    main()
