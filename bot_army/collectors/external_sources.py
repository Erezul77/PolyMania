"""External sources collector for news, macro, and weather."""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import httpx

from ..analyzers.sentiment import SentimentAnalyzer

logger = logging.getLogger("collectors.external_sources")


@dataclass
class WeatherLocation:
    name: str
    lat: float
    lon: float


class ExternalSourcesCollector:
    """Collects outside-world signals and normalizes into external_signals."""

    def __init__(
        self,
        db_manager: Any,
        sentiment_analyzer: SentimentAnalyzer,
        newsapi_key: str = "",
        news_keywords: str = "",
        fred_api_key: str = "",
        fred_series: str = "",
        weather_locations: str = "",
        min_interval_seconds: int = 600,
    ):
        self.db = db_manager
        self.sentiment_analyzer = sentiment_analyzer
        self.newsapi_key = newsapi_key
        self.news_keywords = news_keywords
        self.fred_api_key = fred_api_key
        self.fred_series = fred_series
        self.weather_locations = weather_locations
        self.min_interval_seconds = min_interval_seconds
        self._last_run: Optional[datetime] = None

    async def poll(self) -> int:
        """Fetch external signals if interval has elapsed."""
        if self._last_run and (datetime.utcnow() - self._last_run).total_seconds() < self.min_interval_seconds:
            return 0

        self._last_run = datetime.utcnow()
        signals: List[Dict[str, Any]] = []

        try:
            signals.extend(await self._fetch_news())
        except Exception as exc:
            logger.debug(f"News fetch failed: {exc}")

        try:
            signals.extend(await self._fetch_weather())
        except Exception as exc:
            logger.debug(f"Weather fetch failed: {exc}")

        try:
            signals.extend(await self._fetch_fred())
        except Exception as exc:
            logger.debug(f"Macro fetch failed: {exc}")

        if signals:
            await self.db.store_external_signals(signals)
        return len(signals)

    async def _fetch_news(self) -> List[Dict[str, Any]]:
        if not self.newsapi_key or not self.news_keywords:
            return []

        url = "https://newsapi.org/v2/everything"
        params = {
            "q": self.news_keywords,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": 20,
            "from": (datetime.utcnow() - timedelta(hours=6)).isoformat(),
            "apiKey": self.newsapi_key,
        }
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        results: List[Dict[str, Any]] = []
        for article in data.get("articles", []):
            title = article.get("title") or ""
            description = article.get("description") or ""
            text = f"{title}. {description}".strip()
            if not text:
                continue
            sentiment = self.sentiment_analyzer.analyze_text(text, source="newsapi")
            published_at = article.get("publishedAt") or datetime.utcnow().isoformat()
            results.append(
                {
                    "time": _parse_time(published_at),
                    "source": "newsapi",
                    "source_id": article.get("url") or title[:120],
                    "channel": article.get("source", {}).get("name"),
                    "sentiment_score": sentiment.score,
                    "magnitude": sentiment.magnitude,
                    "label": sentiment.label,
                    "text": text,
                    "url": article.get("url"),
                    "metadata": article,
                }
            )
        return results

    async def _fetch_weather(self) -> List[Dict[str, Any]]:
        locations = _parse_weather_locations(self.weather_locations)
        if not locations:
            return []

        results: List[Dict[str, Any]] = []
        async with httpx.AsyncClient(timeout=15) as client:
            for loc in locations:
                params = {
                    "latitude": loc.lat,
                    "longitude": loc.lon,
                    "current": "temperature_2m,precipitation,wind_speed_10m",
                }
                resp = await client.get("https://api.open-meteo.com/v1/forecast", params=params)
                resp.raise_for_status()
                data = resp.json()
                current = data.get("current") or {}
                temp = current.get("temperature_2m")
                precip = current.get("precipitation")
                wind = current.get("wind_speed_10m")
                score = _weather_score(temp, precip, wind)
                results.append(
                    {
                        "time": _parse_time(current.get("time")) if current.get("time") else datetime.utcnow(),
                        "source": "weather",
                        "source_id": f"{loc.name}-{current.get('time')}",
                        "channel": loc.name,
                        "sentiment_score": score,
                        "magnitude": abs(score),
                        "label": "POSITIVE" if score > 0.2 else "NEGATIVE" if score < -0.2 else "NEUTRAL",
                        "text": f"Weather {loc.name}: temp={temp}, precip={precip}, wind={wind}",
                        "metadata": {"location": loc.name, "temperature": temp, "precipitation": precip, "wind_speed": wind},
                    }
                )
        return results

    async def _fetch_fred(self) -> List[Dict[str, Any]]:
        series = [s.strip() for s in self.fred_series.split(",") if s.strip()]
        if not self.fred_api_key or not series:
            return []

        results: List[Dict[str, Any]] = []
        async with httpx.AsyncClient(timeout=15) as client:
            for series_id in series:
                params = {
                    "series_id": series_id,
                    "api_key": self.fred_api_key,
                    "sort_order": "desc",
                    "limit": 2,
                    "file_type": "json",
                }
                resp = await client.get("https://api.stlouisfed.org/fred/series/observations", params=params)
                resp.raise_for_status()
                data = resp.json()
                observations = data.get("observations", [])
                if len(observations) < 2:
                    continue
                latest = observations[0]
                prev = observations[1]
                latest_val = _parse_float(latest.get("value"))
                prev_val = _parse_float(prev.get("value"))
                if latest_val is None or prev_val is None or prev_val == 0:
                    continue
                pct_change = (latest_val - prev_val) / abs(prev_val)
                score = max(-1.0, min(1.0, pct_change * 5))
                results.append(
                    {
                        "time": _parse_time(latest.get("date")),
                        "source": "macro",
                        "source_id": f"{series_id}-{latest.get('date')}",
                        "channel": series_id,
                        "sentiment_score": score,
                        "magnitude": abs(score),
                        "label": "POSITIVE" if score > 0.2 else "NEGATIVE" if score < -0.2 else "NEUTRAL",
                        "text": f"{series_id} change {pct_change:.4f}",
                        "metadata": {"series_id": series_id, "latest": latest_val, "previous": prev_val},
                    }
                )
        return results


def _parse_weather_locations(raw: str) -> List[WeatherLocation]:
    locations: List[WeatherLocation] = []
    if not raw:
        return locations
    for entry in raw.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        name, coords = entry.split(":", 1) if ":" in entry else ("location", entry)
        parts = [p.strip() for p in coords.split(",")]
        if len(parts) != 2:
            continue
        try:
            lat = float(parts[0])
            lon = float(parts[1])
            locations.append(WeatherLocation(name=name.strip() or "location", lat=lat, lon=lon))
        except ValueError:
            continue
    return locations


def _weather_score(temp: Optional[float], precip: Optional[float], wind: Optional[float]) -> float:
    temp_score = 0.0 if temp is None else max(-1.0, min(1.0, (temp - 20.0) / 20.0))
    precip_score = 0.0 if precip is None else -max(0.0, min(1.0, precip / 10.0))
    wind_score = 0.0 if wind is None else -max(0.0, min(1.0, wind / 20.0))
    score = (temp_score + precip_score + wind_score) / 3.0
    return max(-1.0, min(1.0, score))


def _parse_float(value: Optional[str]) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_time(value: Optional[str]) -> datetime:
    if not value:
        return datetime.utcnow()
    try:
        if value.endswith("Z"):
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        return datetime.fromisoformat(value)
    except Exception:
        return datetime.utcnow()
