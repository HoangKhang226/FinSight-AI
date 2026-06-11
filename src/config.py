"""
FinSight AI — Configuration Module
Đọc setting.yaml + logging.yaml, expose ra config singleton dùng chung.
"""

import os
import yaml
import logging
import logging.config
from pathlib import Path
from typing import Any, Optional


# ── Xác định đường dẫn gốc ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
SETTING_PATH = CONFIG_DIR / "setting.yaml"
LOGGING_PATH = CONFIG_DIR / "logging.yaml"


def _load_yaml(path: Path) -> dict:
    """Đọc file YAML, trả dict rỗng nếu file không tồn tại."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def setup_logging() -> None:
    """Khởi tạo logging từ config/logging.yaml."""
    if not LOGGING_PATH.exists():
        logging.basicConfig(level=logging.INFO)
        logging.warning(f"Logging config not found at {LOGGING_PATH}, using basicConfig.")
        return

    log_cfg = _load_yaml(LOGGING_PATH)

    # Đảm bảo thư mục logs/ tồn tại
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    # Chỉnh đường dẫn file handler thành absolute path
    for handler in log_cfg.get("handlers", {}).values():
        if "filename" in handler:
            handler["filename"] = str(PROJECT_ROOT / handler["filename"])

    logging.config.dictConfig(log_cfg)


class Settings:
    """
    Singleton đọc config/setting.yaml một lần duy nhất.
    Truy cập nested keys bằng dot-notation: settings.get("llm.ollama.model")
    """

    _instance: Optional["Settings"] = None
    _data: dict = {}

    def __new__(cls) -> "Settings":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._data = _load_yaml(SETTING_PATH)
        return cls._instance

    # ── Truy cập nested key ──────────────────────────────────────────────
    def get(self, dotted_key: str, default: Any = None) -> Any:
        """
        Truy cập nested key dạng 'llm.ollama.model'.
        Trả default nếu không tìm thấy.
        """
        keys = dotted_key.split(".")
        node = self._data
        for k in keys:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                return default
        return node

    # ── Shortcut properties ──────────────────────────────────────────────
    @property
    def app_name(self) -> str:
        return self.get("app.project_name", "FinSight-AI")

    @property
    def debug(self) -> bool:
        return self.get("app.debug", False)

    @property
    def database_url(self) -> str:
        return self.get("storage.database.url", "sqlite:///./storage/app_db.sqlite")

    @property
    def database_echo(self) -> bool:
        return self.get("storage.database.echo", False)

    @property
    def ollama_base_url(self) -> str:
        return self.get("llm.ollama.base_url", "http://localhost:11434")

    @property
    def ollama_model(self) -> str:
        return self.get("llm.ollama.model", "qwen2.5:7b")

    @property
    def vlm_model(self) -> str:
        return self.get("llm.ollama.vlm_model", "qwen2-vl:2b")

    @property
    def embed_model(self) -> str:
        return self.get("llm.ollama.embed_model", "nomic-embed-text")

    def __repr__(self) -> str:
        return f"<Settings app={self.app_name} debug={self.debug}>"


# ── Module-level convenience ─────────────────────────────────────────────
settings = Settings()


def get_logger(name: str = "finsight") -> logging.Logger:
    """Lấy logger đã cấu hình. Gọi setup_logging() nếu chưa khởi tạo."""
    return logging.getLogger(name)
