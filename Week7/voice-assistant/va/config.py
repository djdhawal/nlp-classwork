"""Load config.yaml and set up logging. Every setting lives in the config file."""
from __future__ import annotations

import copy
import logging
import os
import sys
import logging.handlers
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_DIR / "config.yaml"


def on_colab() -> bool:
    return "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ


def load_config(path: str | Path | None = None, **overrides) -> dict:
    """Read the YAML config. `overrides` are dotted keys, e.g. {"llm.model": "qwen3:8b"}.

    On Google Colab the `colab:` section of the config is applied first.
    """
    path = Path(path or DEFAULT_CONFIG).resolve()
    cfg = yaml.safe_load(path.read_text())
    cfg["_dir"] = str(path.parent)
    cfg["_colab"] = on_colab()
    if cfg["_colab"]:
        for dotted, value in (cfg.get("colab") or {}).items():
            set_key(cfg, dotted, value)
    for dotted, value in overrides.items():
        set_key(cfg, dotted, value)
    return cfg


def set_key(cfg: dict, dotted: str, value) -> None:
    node = cfg
    *parents, last = dotted.split(".")
    for key in parents:
        node = node.setdefault(key, {})
    node[last] = value


def with_overrides(cfg: dict, **overrides) -> dict:
    """A copy of cfg with dotted-key overrides applied."""
    new = copy.deepcopy(cfg)
    for dotted, value in overrides.items():
        set_key(new, dotted, value)
    return new


def resolve_path(cfg: dict, relative: str | None) -> Path | None:
    """Paths in the config are relative to the config file's folder."""
    if relative is None:
        return None
    p = Path(relative)
    return p if p.is_absolute() else Path(cfg["_dir"]) / p


def setup_logging(cfg: dict) -> logging.Logger:
    """Rotating log file plus console. Safe to call more than once."""
    log = logging.getLogger("va")
    if getattr(log, "_va_configured", False):
        return log
    level = getattr(logging, str(cfg["logging"].get("level", "INFO")).upper(), logging.INFO)
    log.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(threadName)-10s %(message)s")
    path = resolve_path(cfg, cfg["logging"]["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(path, maxBytes=1_000_000, backupCount=3)
    fh.setFormatter(fmt)
    log.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.WARNING)
    log.addHandler(sh)
    log._va_configured = True
    return log
