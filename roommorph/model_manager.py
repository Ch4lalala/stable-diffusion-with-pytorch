from __future__ import annotations

import inspect
import threading
from pathlib import Path

import torch
from transformers import CLIPTokenizer

from .config import (
    IDLE_DEVICE,
    LOGGER,
    MERGES_FILE,
    MODEL_FILE,
    SKIP_MODEL_LOAD,
    VOCAB_FILE,
)
from .exceptions import ModelFilesMissingError

import model_loader


def missing_model_files() -> list[Path]:
    return [path for path in (MODEL_FILE, VOCAB_FILE, MERGES_FILE) if not path.is_file()]


def missing_files_message(missing: list[Path]) -> str:
    expected = "\n".join(f"- {path}" for path in missing)
    return (
        "File model/tokenizer belum lengkap. Pastikan file berikut tersedia:\n"
        f"{expected}\n"
        "Checkpoint dapat diarahkan dengan SD_MODEL_PATH."
    )


def _build_tokenizer() -> CLIPTokenizer:
    parameters = inspect.signature(CLIPTokenizer.__init__).parameters
    if "vocab" in parameters:
        return CLIPTokenizer(vocab=str(VOCAB_FILE), merges=str(MERGES_FILE))
    return CLIPTokenizer(vocab_file=str(VOCAB_FILE), merges_file=str(MERGES_FILE))


class ModelStore:
    def __init__(self) -> None:
        self._models: dict[str, torch.nn.Module] | None = None
        self._tokenizer: CLIPTokenizer | None = None
        self._load_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._models is not None and self._tokenizer is not None

    def load(self) -> tuple[dict[str, torch.nn.Module], CLIPTokenizer]:
        if SKIP_MODEL_LOAD:
            raise RuntimeError("Model loading dinonaktifkan oleh ROOMMORPH_SKIP_MODEL_LOAD.")

        with self._load_lock:
            if self._models is None or self._tokenizer is None:
                missing = missing_model_files()
                if missing:
                    raise ModelFilesMissingError(missing_files_message(missing))

                LOGGER.info("Loading tokenizer from %s and %s", VOCAB_FILE, MERGES_FILE)
                tokenizer = _build_tokenizer()

                LOGGER.info("Loading Stable Diffusion weights from %s", MODEL_FILE)
                models = model_loader.preload_models_from_standard_weights(
                    str(MODEL_FILE), IDLE_DEVICE
                )
                for model in models.values():
                    model.eval()

                self._tokenizer = tokenizer
                self._models = models
                LOGGER.info(
                    "Stable Diffusion components loaded on idle device %s", IDLE_DEVICE
                )

        return self._models, self._tokenizer

    def offload(self) -> None:
        if self._models is None:
            return

        for name, model in self._models.items():
            try:
                model.to(IDLE_DEVICE)
            except Exception:
                LOGGER.exception("Could not offload model component %s", name)


MODEL_STORE = ModelStore()


def initial_status() -> str:
    if SKIP_MODEL_LOAD:
        return (
            "Smoke-test mode is active (ROOMMORPH_SKIP_MODEL_LOAD=1). "
            "The interface works, but inference is disabled and no AI image will be returned."
        )

    missing = missing_model_files()
    if missing:
        return missing_files_message(missing)

    return (
        "Model files found. The model is loaded once on the first generation, "
        "so the first request has additional startup time."
    )
