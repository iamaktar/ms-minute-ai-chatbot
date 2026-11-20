"""Configuration management for the Ms. Minute AI Chatbot application."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


class ConfigurationError(ValueError):
    """Raised when required configuration values are missing or invalid."""


@dataclass(slots=True)
class Settings:
    """Application settings loaded from environment variables."""

    api_key: str
    model_name: str = "gemini-3.5-flash-lite"
    temperature: float = 0.7
    system_instruction: str = (
        "You are Ms. Minute, a sharp, helpful, and friendly AI assistant. "
        "Keep responses clear, concise, and practical."
    )
    max_history_messages: int = 20
    request_timeout_seconds: int = 60

    @classmethod
    def from_env(cls, env_path: Path | None = None) -> Settings:
        """Load and validate settings from a .env file and system environment.

        Args:
            env_path: Optional path to a specific .env file.

        Returns:
            A validated Settings instance.

        Raises:
            ConfigurationError: If required variables are missing or malformed.
        """
        load_dotenv(dotenv_path=env_path)

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key or api_key == "your_api_key_here":
            raise ConfigurationError(
                "Missing GEMINI_API_KEY. Copy .env.example to .env and set a valid API key."
            )

        model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
        if not model_name:
            raise ConfigurationError("GEMINI_MODEL cannot be empty.")

        temperature = _parse_float(
            name="GEMINI_TEMPERATURE",
            default=0.7,
            min_value=0.0,
            max_value=2.0,
        )

        max_history_messages = _parse_int(
            name="MAX_HISTORY_MESSAGES",
            default=20,
            min_value=2,
        )

        request_timeout_seconds = _parse_int(
            name="REQUEST_TIMEOUT_SECONDS",
            default=60,
            min_value=5,
        )

        system_instruction = os.getenv(
            "GEMINI_SYSTEM_INSTRUCTION",
            "You are Ms. Minute, a sharp, helpful, and friendly AI assistant. "
            "Keep responses clear, concise, and practical.",
        ).strip()

        return cls(
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            system_instruction=system_instruction,
            max_history_messages=max_history_messages,
            request_timeout_seconds=request_timeout_seconds,
        )


def _parse_float(
    name: str,
    default: float,
    min_value: float,
    max_value: float,
) -> float:
    """Parse and validate a float environment variable."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw.strip())
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a valid number, got: {raw!r}") from exc
    if not (min_value <= value <= max_value):
        raise ConfigurationError(
            f"{name} must be between {min_value} and {max_value}, got: {value}"
        )
    return value


def _parse_int(name: str, default: int, min_value: int) -> int:
    """Parse and validate an integer environment variable."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer, got: {raw!r}") from exc
    if value < min_value:
        raise ConfigurationError(
            f"{name} must be at least {min_value}, got: {value}"
        )
    return value