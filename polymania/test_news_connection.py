import sys

from .config import settings
from .news_client import search_news
from .notifier import send_telegram_message


def main() -> None:
    """
    Simple manual test for News API + Telegram.

    It will:
    - check that NEWS_API_KEY is configured
    - query a fixed keyword (e.g. 'polymarket')
    - print results to console
    - send a short summary to Telegram (if Telegram is configured)
    """
    print("=== PolyMania News Test ===")
    print(f"NEWS_API_KEY set: {bool(settings.news_api_key)}")
    print(f"NEWS_LANGUAGE: {settings.news_language!r}")
    print()

    if not settings.news_api_key:
        print("ERROR: NEWS_API_KEY is missing in .env")
        sys.exit(1)

    query = "polymarket"
    print(f"Querying news for: {query!r}")
    articles = search_news(query, max_results=3)

    if not articles:
        print("No articles returned (empty result or API error).")
        sys.exit(1)

    print(f"Got {len(articles)} article(s):")
    for i, a in enumerate(articles, start=1):
        title = a.get("title") or "Untitled"
        source = a.get("source") or "Unknown"
        url = a.get("url") or ""
        print(f"{i}. {title} — {source}")
        if url:
            print(f"   {url}")

    # Build a Telegram message (if Telegram is configured)
    if settings.telegram_bot_token and settings.telegram_chat_id:
        lines = []
        lines.append("📰 *PolyMania News Test*")
        lines.append("")
        lines.append(f"Top {len(articles)} headlines for query: `{query}`")
        lines.append("")
        for i, a in enumerate(articles, start=1):
            title = a.get("title") or "Untitled"
            source = a.get("source") or "Unknown"
            url = a.get("url") or ""
            lines.append(f"{i}. {title} — _{source}_")
            if url:
                lines.append(url)

        text = "\n".join(lines)
        ok = send_telegram_message(text)
        if ok:
            print("\nSUCCESS: News summary sent to Telegram.")
        else:
            print("\nWARNING: Could not send news summary to Telegram (see error above).")
    else:
        print("\nNOTE: Telegram not configured, skipping Telegram send.")

    sys.exit(0)


if __name__ == "__main__":
    main()

