"""Shared LLM configuration loaded from the project-root .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"


def load_env_file(path: str | Path = DEFAULT_ENV_FILE, *, override: bool = False) -> Path:
    """Load simple KEY=VALUE entries without requiring python-dotenv."""
    env_path = Path(path).expanduser().resolve()
    if not env_path.is_file():
        return env_path

    for line_number, raw_line in enumerate(env_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"Invalid .env entry at {env_path}:{line_number}")
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if not name:
            raise ValueError(f"Empty variable name at {env_path}:{line_number}")
        if override or name not in os.environ:
            os.environ[name] = value
    return env_path


@dataclass(frozen=True)
class LLMConfig:
    host: str
    key: str
    model: str
    timeout: int


def get_llm_config(*, timeout: int | None = None, env_file: str | Path = DEFAULT_ENV_FILE) -> LLMConfig:
    """Return validated LLM settings; process variables override .env values."""
    path = load_env_file(env_file)
    host = os.getenv("LLM4AD_API_HOST", "").strip()
    key = os.getenv("LLM4AD_API_KEY", "").strip()
    model = os.getenv("LLM4AD_MODEL", "").strip()
    configured_timeout = os.getenv("LLM4AD_TIMEOUT", "240").strip()
    if not host or not key or not model:
        missing = [
            name
            for name, value in (
                ("LLM4AD_API_HOST", host),
                ("LLM4AD_API_KEY", key),
                ("LLM4AD_MODEL", model),
            )
            if not value
        ]
        raise RuntimeError(f"Missing {', '.join(missing)} in environment or {path}")
    return LLMConfig(
        host=host,
        key=key,
        model=model,
        timeout=int(timeout if timeout is not None else configured_timeout),
    )

