import asyncio

from telethon.tl.types import Channel, Chat, User

from .telegram_user_client import get_telegram_user_client


async def main() -> None:
    """
    List dialogs (chats, groups, channels) for the logged-in user.

    Requirements:
    - You must have run telegram_first_login.py at least once (session file exists).
    - The account must have joined / subscribed to the relevant channels.

    Output:
    - For each dialog, prints:
      - type: private / group / supergroup / channel
      - title / name
      - id
      - username (if available)
    """
    client = get_telegram_user_client()
    print("=== PolyMania Telegram dialogs ===")

    async with client:
        async for dialog in client.iter_dialogs():
            entity = dialog.entity
            name = dialog.name
            chat_id = entity.id
            username = getattr(entity, "username", None)

            if isinstance(entity, User):
                chat_type = "private"
            elif isinstance(entity, Chat):
                chat_type = "group"
            elif isinstance(entity, Channel):
                if entity.megagroup:
                    chat_type = "supergroup"
                else:
                    chat_type = "channel"
            else:
                chat_type = type(entity).__name__

            print("---- DIALOG ----")
            print(f"type:     {chat_type}")
            print(f"name:     {name!r}")
            print(f"id:       {chat_id}")
            print(f"username: {username!r}")
            print()

    print("Done. Pick the channels/groups you care about by id or username.")


if __name__ == "__main__":
    asyncio.run(main())
