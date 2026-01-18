import os
from typing import Annotated, Optional

from dotenv import load_dotenv
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load variables from .env file if present
load_dotenv()

DEFAULT_WAR_KEYWORDS = [
    "gaza",
    "israel",
    "lebanon",
    "hezbollah",
    "hamas",
    "iran",
    "strike",
    "strikes",
    "offensive",
    "ground offensive",
    "invade",
    "invasion",
    "ceasefire",
    "airstrike",
    "rocket",
    "border",
]


def _parse_keywords(raw: Optional[str]) -> list[str]:
    if not raw:
        return []
    # Replace newlines with spaces first (in case .env has multi-line values)
    raw = raw.replace('\n', ' ').replace('\r', ' ')
    return [part.strip().lower() for part in raw.split(",") if part.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=False,
        extra='ignore',
    )

    # Polymarket endpoints
    gamma_base: str = Field(default="https://gamma-api.polymarket.com", env="POLYMARKET_GAMMA_BASE")
    data_base: str = Field(default="https://data-api.polymarket.com", env="POLYMARKET_DATA_BASE")

    # Detection parameters (can be overridden via env)
    poll_interval_sec: int = Field(default=15, env="POLL_INTERVAL_SEC")
    recent_window_sec: int = Field(default=60, env="RECENT_WINDOW_SEC")
    base_window_sec: int = Field(default=600, env="BASE_WINDOW_SEC")
    min_price_jump: float = Field(default=0.12, env="MIN_PRICE_JUMP")
    min_recent_volume: float = Field(default=200.0, env="MIN_RECENT_VOLUME")
    min_recent_trades: int = Field(default=5, env="MIN_RECENT_TRADES")
    dominance_threshold: float = Field(default=0.8, env="DOMINANCE_THRESHOLD")

    # Optional filters
    max_event_age_hours: Optional[int] = Field(default=6, env="MAX_EVENT_AGE_HOURS")
    
    # Keyword filters - stored as strings in .env, converted to lists via properties
    watch_keywords_raw: str = Field(default="", validation_alias="WATCH_KEYWORDS")
    war_keywords_raw: str = Field(default="", validation_alias="WAR_KEYWORDS")
    
    max_event_age_hours_war: Optional[int] = Field(default=None, env="MAX_EVENT_AGE_HOURS_WAR")

    # Correlation / cluster meta-signals
    corr_enabled: bool = Field(default=True, env="CORR_ENABLED")
    corr_window_sec: int = Field(default=600, env="CORR_WINDOW_SEC")
    corr_min_signals_per_topic: int = Field(default=2, env="CORR_MIN_SIGNALS_PER_TOPIC")
    corr_cooldown_sec: int = Field(default=600, env="CORR_COOLDOWN_SEC")

    # Cooldown between alerts per event (seconds)
    cooldown_sec: int = Field(default=300, env="COOLDOWN_SEC")

    # Logging
    log_file: str = Field(default="polymania.log", env="LOG_FILE")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # Telegram alert config
    telegram_bot_token: Optional[str] = Field(default=None, env="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = Field(default=None, env="TELEGRAM_CHAT_ID")

    # News API (e.g. newsapi.org)
    news_api_key: Optional[str] = Field(default=None, env="NEWS_API_KEY")
    news_language: str = Field(default="en", env="NEWS_LANGUAGE")

    # Telegram user client settings (Telethon)
    tg_api_id: Optional[int] = Field(default=None, env="TG_API_ID")
    tg_api_hash: Optional[str] = Field(default=None, env="TG_API_HASH")
    tg_session_name: str = Field(default="polymania_session", env="TG_SESSION_NAME")

    @property
    def watch_keywords(self) -> list[str]:
        """Parse watch keywords from raw string."""
        if not self.watch_keywords_raw:
            return []
        return _parse_keywords(self.watch_keywords_raw)

    @property
    def war_keywords(self) -> list[str]:
        """Parse war keywords from raw string, with defaults."""
        if not self.war_keywords_raw:
            return DEFAULT_WAR_KEYWORDS.copy()
        return _parse_keywords(self.war_keywords_raw)

    @field_validator("max_event_age_hours", "max_event_age_hours_war", mode="before")
    @classmethod
    def _empty_str_to_none(cls, value):
        if value is None:
            return value
        if isinstance(value, str) and value.strip() == "":
            return None
        return value


def load_settings() -> Settings:
    return Settings()


settings = load_settings()
