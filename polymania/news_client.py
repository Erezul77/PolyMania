from typing import Any, Dict, List

import requests

from .config import settings


def search_news(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """
    Query recent news for the event using a 3rd-party news API.
    
    This implementation uses newsapi.org as an example.
    You must provide NEWS_API_KEY in your .env file.
    
    If no API key is configured, returns an empty list.
    """
    if not settings.news_api_key:
        return []

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": settings.news_language,
        "pageSize": max_results,
        "sortBy": "publishedAt",
        "apiKey": settings.news_api_key,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
    except Exception:
        return []

    data = resp.json()
    articles = data.get("articles", [])
    results: List[Dict[str, Any]] = []

    for a in articles[:max_results]:
        results.append(
            {
                "title": a.get("title"),
                "source": a.get("source", {}).get("name"),
                "url": a.get("url"),
                "publishedAt": a.get("publishedAt"),
            }
        )

    return results

