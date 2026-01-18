import sys

from .config import settings
from .notifier import send_telegram_message


def main() -> None:
    """
    Simple manual test for Telegram configuration.

    It will:
    - print which settings are loaded
    - try to send a test message to the configured chat
    - print whether it succeeded
    """
    print("=== PolyMania Telegram Test ===")
    print(f"TELEGRAM_BOT_TOKEN set: {bool(settings.telegram_bot_token)}")
    print(f"TELEGRAM_CHAT_ID: {settings.telegram_chat_id!r}")
    print()

    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        print("ERROR: Telegram bot token or chat id are missing in .env")
        sys.exit(1)

    msg = "✅ PolyMania test message: Telegram configuration looks good!"

    ok = send_telegram_message(msg)
    if ok:
        print("SUCCESS: Test message was sent. Check your Telegram.")
        sys.exit(0)
    else:
        print("FAILED: Could not send Telegram message. See error above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

