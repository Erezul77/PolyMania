import sys
from typing import Any, Dict, List, Optional

import requests

from .config import settings


def main() -> None:
    """
    List all chats that the bot has seen via getUpdates.

    Steps for the user:
    1. Add your bot to the target group (with your friend).
    2. Send a message in that group (e.g. 'poly test').
    3. Run this script (via the VSCode/Cursor task).
    4. Look for a line where 'chat_type' is 'group' or 'supergroup'
       and 'title' matches your group's name.
    5. Copy the 'chat_id' from that line into TELEGRAM_CHAT_ID in .env.
    """
    print("=== PolyMania Telegram Chats List ===")
    print(f"TELEGRAM_BOT_TOKEN set: {bool(settings.telegram_bot_token)}")
    print()

    if not settings.telegram_bot_token:
        print("ERROR: TELEGRAM_BOT_TOKEN is missing in .env")
        sys.exit(1)

    url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/getUpdates"

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print("ERROR: Failed to call getUpdates:", e)
        sys.exit(1)

    data = resp.json()
    results: List[Dict[str, Any]] = data.get("result", [])
    if not results:
        print("No updates found. Make sure you've:")
        print("1) Started a chat with the bot")
        print("2) Added the bot to the group and sent a message there")
        sys.exit(0)

    print(f"Found {len(results)} update(s). Listing distinct chats:\n")

    seen_ids = set()

    for upd in results:
        msg: Optional[Dict[str, Any]] = upd.get("message") or upd.get("channel_post")
        if not msg:
            continue

        chat = msg.get("chat", {})
        chat_id = chat.get("id")
        if chat_id in seen_ids:
            continue
        seen_ids.add(chat_id)

        chat_type = chat.get("type")
        title = chat.get("title")
        username = chat.get("username")

        print("---- CHAT ----")
        print(f"chat_id:   {chat_id}")
        print(f"type:      {chat_type}")
        print(f"title:     {title!r}")
        print(f"username:  {username!r}")
        print()

    print("Pick the relevant chat_id (group or private) and put it in TELEGRAM_CHAT_ID in your .env.")
    sys.exit(0)


if __name__ == "__main__":
    main()

