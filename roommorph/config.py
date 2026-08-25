from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parent.parent
SD_DIR = ROOT_DIR / "sd"
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"

# The educational SD modules intentionally use flat imports for the notebook.
if str(SD_DIR) not in sys.path:
    sys.path.insert(0, str(SD_DIR))

logging.basicConfig(
    level=os.getenv("ROOMMORPH_LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("roommorph")

SUPPORTED_RESOLUTIONS = (256, 512)
MIN_STEPS = 5
MAX_STEPS = 50
MIN_STRENGTH = 0.20
MAX_STRENGTH = 0.90
MIN_CFG_SCALE = 1.0
MAX_CFG_SCALE = 12.0
MAX_SEED = 2**31 - 1


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _resolve_configured_path(env_name: str, default: Path) -> Path:
    configured = os.getenv(env_name)
    path = Path(configured).expanduser() if configured else default
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path.resolve(strict=False)


MODEL_FILE = _resolve_configured_path(
    "SD_MODEL_PATH", DATA_DIR / "v1-5-pruned-emaonly.ckpt"
)
VOCAB_FILE = _resolve_configured_path("SD_VOCAB_PATH", DATA_DIR / "vocab.json")
MERGES_FILE = _resolve_configured_path("SD_MERGES_PATH", DATA_DIR / "merges.txt")
SKIP_MODEL_LOAD = _env_flag("ROOMMORPH_SKIP_MODEL_LOAD")


def select_runtime_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")

    return torch.device("cpu")


RUN_DEVICE = select_runtime_device()
IDLE_DEVICE = torch.device("cpu")
