"""External signals collector for news/telegram/etc."""

import csv
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..analyzers.sentiment import SentimentAnalyzer

logger = logging.getLogger("collectors.external_signals")


@dataclass
class ExternalSignalRecord:
    """Normalized external signal record."""
    time: datetime
    source: str
    source_id: str
    channel: Optional[str] = None
    market_id: Optional[str] = None
    event_id: Optional[str] = None
    sentiment_score: Optional[float] = None
    magnitude: Optional[float] = None
    label: Optional[str] = None
    text: Optional[str] = None
    url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time": self.time,
            "source": self.source,
            "source_id": self.source_id,
            "channel": self.channel,
            "market_id": self.market_id,
            "event_id": self.event_id,
            "sentiment_score": self.sentiment_score,
            "magnitude": self.magnitude,
            "label": self.label,
            "text": self.text,
            "url": self.url,
            "metadata": self.metadata,
        }


class ExternalSignalsCollector:
    """
    Collects external signals from CSV artifacts (news/telegram exports).
    """

    def __init__(
        self,
        paths: List[str],
        sentiment_analyzer: SentimentAnalyzer,
        db_manager: Any,
    ):
        self.paths = [Path(p).expanduser() for p in paths if p]
        self.sentiment_analyzer = sentiment_analyzer
        self.db = db_manager
        self._last_row_index: Dict[Path, int] = {}

    async def poll(self) -> int:
        """Read new external rows and store them."""
        total = 0
        for path in self.paths:
            if not path.exists():
                continue
            rows = list(self._read_csv(path))
            if not rows:
                continue
            last_idx = self._last_row_index.get(path, 0)
            new_rows = rows[last_idx:]
            if not new_rows:
                continue
            records = self._parse_rows(path.name.lower(), new_rows)
            if records:
                await self.db.store_external_signals([r.to_dict() for r in records])
                total += len(records)
            self._last_row_index[path] = len(rows)
        return total

    def _read_csv(self, path: Path):
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    yield row
        except Exception as exc:
            logger.debug(f"Failed reading {path}: {exc}")

    def _parse_rows(self, filename: str, rows: List[Dict[str, str]]) -> List[ExternalSignalRecord]:
        records: List[ExternalSignalRecord] = []
        if "telegram" in filename:
            records.extend(self._parse_telegram(rows))
        elif "signals" in filename:
            records.extend(self._parse_signals(rows))
        else:
            records.extend(self._parse_generic(rows))
        return records

    def _parse_telegram(self, rows: List[Dict[str, str]]) -> List[ExternalSignalRecord]:
        records: List[ExternalSignalRecord] = []
        for row in rows:
            text = (row.get("text_excerpt") or "").strip()
            if not text:
                continue
            ts = self._parse_time(row.get("message_date_utc"))
            source_id = row.get("message_id") or f"{row.get('message_date_utc','')}-{hash(text)}"
            channel = row.get("channel_username") or row.get("channel_title")
            sentiment = self.sentiment_analyzer.analyze_text(text, source="telegram")
            records.append(
                ExternalSignalRecord(
                    time=ts,
                    source="telegram",
                    source_id=str(source_id),
                    channel=channel,
                    sentiment_score=sentiment.score,
                    magnitude=sentiment.magnitude,
                    label=sentiment.label,
                    text=text,
                    url=row.get("message_url"),
                    metadata=row,
                )
            )
        return records

    def _parse_signals(self, rows: List[Dict[str, str]]) -> List[ExternalSignalRecord]:
        records: List[ExternalSignalRecord] = []
        for row in rows:
            news_titles = (row.get("news_titles") or "").strip()
            event_title = (row.get("event_title") or "").strip()
            signal_reason = (row.get("signal_reason") or "").strip()
            text = " ".join(part for part in [event_title, signal_reason, news_titles] if part)
            if not text:
                continue
            ts = self._parse_time(row.get("timestamp_iso"))
            event_id = row.get("event_id") or None
            source_id = f"{event_id}-{row.get('timestamp_iso','')}-{row.get('signal_type','')}"
            sentiment = self.sentiment_analyzer.analyze_text(text, source="news")
            records.append(
                ExternalSignalRecord(
                    time=ts,
                    source="news",
                    source_id=source_id,
                    channel="signals_csv",
                    event_id=event_id,
                    sentiment_score=sentiment.score,
                    magnitude=sentiment.magnitude,
                    label=sentiment.label,
                    text=text,
                    metadata=row,
                )
            )
        return records

    def _parse_generic(self, rows: List[Dict[str, str]]) -> List[ExternalSignalRecord]:
        records: List[ExternalSignalRecord] = []
        for row in rows:
            text = (row.get("text") or row.get("message") or "").strip()
            if not text:
                continue
            ts = self._parse_time(row.get("time") or row.get("timestamp"))
            source = row.get("source") or "external"
            source_id = row.get("id") or f"{ts.isoformat()}-{hash(text)}"
            sentiment = self.sentiment_analyzer.analyze_text(text, source=source)
            records.append(
                ExternalSignalRecord(
                    time=ts,
                    source=source,
                    source_id=str(source_id),
                    channel=row.get("channel"),
                    market_id=row.get("market_id"),
                    event_id=row.get("event_id"),
                    sentiment_score=sentiment.score,
                    magnitude=sentiment.magnitude,
                    label=sentiment.label,
                    text=text,
                    url=row.get("url"),
                    metadata=row,
                )
            )
        return records

    def _parse_time(self, value: Optional[str]) -> datetime:
        if not value:
            return datetime.utcnow()
        try:
            if value.endswith("Z"):
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            return datetime.fromisoformat(value)
        except Exception:
            return datetime.utcnow()
