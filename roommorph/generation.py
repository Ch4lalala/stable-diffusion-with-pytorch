from __future__ import annotations

import gc
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import torch
from PIL import Image

from . import model_manager
from .config import (
    IDLE_DEVICE,
    LOGGER,
    MAX_CFG_SCALE,
    MAX_SEED,
    MAX_STEPS,
    MAX_STRENGTH,
    MIN_CFG_SCALE,
    MIN_STEPS,
    MIN_STRENGTH,
    RUN_DEVICE,
    SKIP_MODEL_LOAD,
    SUPPORTED_RESOLUTIONS,
)
from .exceptions import ModelFilesMissingError, RoomMorphValidationError
from .image_utils import preprocess_room_image, save_output, to_rgb_pil_image
from .prompts import NEGATIVE_PROMPT, STYLE_PROMPTS, build_prompt, build_text_to_image_prompt

import pipeline as sd_pipeline


@dataclass(frozen=True)
class GenerationSettings:
    style: str
    instruction: str
    strength: float
    steps: int
    do_cfg: bool
    cfg_scale: float
    seed: int
    resolution: int


@dataclass(frozen=True)
class TextToImageSettings:
    style: str
    positive_prompt: str
    negative_prompt: str
    steps: int
    cfg_scale: float
    seed: int
    resolution: int


INFERENCE_LOCK = threading.Lock()


def _clear_device_cache() -> None:
    if RUN_DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    elif RUN_DEVICE.type == "mps":
        mps_module = getattr(torch, "mps", None)
        if mps_module is not None and hasattr(mps_module, "empty_cache"):
            mps_module.empty_cache()
    gc.collect()


def _validated_integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise RoomMorphValidationError(f"{label} harus berupa angka bulat.") from exc

    if not math.isfinite(numeric) or not numeric.is_integer():
        raise RoomMorphValidationError(f"{label} harus berupa angka bulat.")

    result = int(numeric)
    if not minimum <= result <= maximum:
        raise RoomMorphValidationError(
            f"{label} harus berada di antara {minimum} dan {maximum}."
        )
    return result


def _validated_float(
    value: Any, label: str, minimum: float, maximum: float
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RoomMorphValidationError(f"{label} harus berupa angka.") from exc

    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise RoomMorphValidationError(
            f"{label} harus berada di antara {minimum:.2f} dan {maximum:.2f}."
        )
    return result


def validate_settings(
    style: str,
    instruction: str | None,
    strength: Any,
    steps: Any,
    do_cfg: Any,
    cfg_scale: Any,
    seed: Any,
    resolution: Any,
) -> GenerationSettings:
    if style not in STYLE_PROMPTS:
        raise RoomMorphValidationError("Pilih salah satu style interior yang tersedia.")

    cleaned_instruction = " ".join((instruction or "").split())
    if len(cleaned_instruction) > 500:
        raise RoomMorphValidationError("Instruksi tambahan maksimal 500 karakter.")

    validated_resolution = _validated_integer(
        resolution, "Resolusi", min(SUPPORTED_RESOLUTIONS), max(SUPPORTED_RESOLUTIONS)
    )
    if validated_resolution not in SUPPORTED_RESOLUTIONS:
        raise RoomMorphValidationError("Resolusi harus 256 atau 512.")

    return GenerationSettings(
        style=style,
        instruction=cleaned_instruction,
        strength=_validated_float(
            strength, "Strength", MIN_STRENGTH, MAX_STRENGTH
        ),
        steps=_validated_integer(steps, "Inference steps", MIN_STEPS, MAX_STEPS),
        do_cfg=bool(do_cfg),
        cfg_scale=_validated_float(
            cfg_scale, "CFG scale", MIN_CFG_SCALE, MAX_CFG_SCALE
        ),
        seed=_validated_integer(seed, "Seed", 0, MAX_SEED),
        resolution=validated_resolution,
    )


def validate_text_to_image_settings(
    style: str,
    positive_prompt: str | None,
    negative_prompt: str | None,
    resolution: Any,
    steps: Any,
    cfg_scale: Any,
    seed: Any,
) -> TextToImageSettings:
    if style not in STYLE_PROMPTS:
        raise RoomMorphValidationError("Pilih salah satu style interior yang tersedia.")

    cleaned_positive_prompt = " ".join((positive_prompt or "").split())
    cleaned_negative_prompt = " ".join((negative_prompt or "").split())
    if not cleaned_positive_prompt:
        raise RoomMorphValidationError("Positive prompt tidak boleh kosong.")
    if len(cleaned_positive_prompt) > 500:
        raise RoomMorphValidationError("Positive prompt maksimal 500 karakter.")
    if len(cleaned_negative_prompt) > 500:
        raise RoomMorphValidationError("Negative prompt maksimal 500 karakter.")

    validated_resolution = _validated_integer(
        resolution, "Resolusi", min(SUPPORTED_RESOLUTIONS), max(SUPPORTED_RESOLUTIONS)
    )
    if validated_resolution not in SUPPORTED_RESOLUTIONS:
        raise RoomMorphValidationError("Resolusi harus 256 atau 512.")

    return TextToImageSettings(
        style=style,
        positive_prompt=cleaned_positive_prompt,
        negative_prompt=cleaned_negative_prompt,
        steps=_validated_integer(steps, "Inference steps", MIN_STEPS, MAX_STEPS),
        cfg_scale=_validated_float(
            cfg_scale, "CFG scale", MIN_CFG_SCALE, MAX_CFG_SCALE
        ),
        seed=_validated_integer(seed, "Seed", 0, MAX_SEED),
        resolution=validated_resolution,
    )


def _configure_pipeline_resolution(resolution: int) -> None:
    sd_pipeline.WIDTH = resolution
    sd_pipeline.HEIGHT = resolution
    sd_pipeline.LATENTS_WIDTH = resolution // 8
    sd_pipeline.LATENTS_HEIGHT = resolution // 8


def _metadata(
    settings: GenerationSettings,
    elapsed_seconds: float,
    output_path: Path,
) -> str:
    cfg_value = (
        f"enabled (scale {settings.cfg_scale:.1f})" if settings.do_cfg else "disabled"
    )
    approximate_steps = settings.steps * settings.strength
    return "\n".join(
        [
            "Status: generation complete",
            f"Runtime device: {RUN_DEVICE.type}",
            f"Resolution: {settings.resolution} x {settings.resolution}",
            f"Inference steps: {settings.steps}",
            f"Approximate img2img steps: {approximate_steps:.1f}",
            f"Strength: {settings.strength:.2f}",
            f"CFG: {cfg_value}",
            f"Seed: {settings.seed}",
            "Sampler: DDPM",
            f"Elapsed generation time: {elapsed_seconds:.2f} seconds",
            f"Saved output: {output_path.name}",
        ]
    )


def _text_to_image_metadata(
    settings: TextToImageSettings,
    effective_prompt: str,
    elapsed_seconds: float,
    output_path: Path,
) -> str:
    negative_prompt = settings.negative_prompt or "(none)"
    return "\n".join(
        [
            "Status: generation complete",
            "Mode: text-to-image",
            f"Runtime device: {RUN_DEVICE.type}",
            f"Style preset: {settings.style}",
            f"Effective prompt: {effective_prompt}",
            f"Negative prompt: {negative_prompt}",
            f"Resolution: {settings.resolution} x {settings.resolution}",
            f"Inference steps: {settings.steps}",
            f"CFG scale: {settings.cfg_scale:.1f}",
            f"Seed: {settings.seed}",
            "Sampler: DDPM",
            f"Elapsed generation time: {elapsed_seconds:.2f} seconds",
            f"Saved output: {output_path.name}",
        ]
    )


def generate_design(
    input_image: Image.Image | np.ndarray | str | Path | None,
    style: str,
    instruction: str,
    strength: float,
    steps: int,
    do_cfg: bool,
    cfg_scale: float,
    seed: int,
    resolution: int,
) -> tuple[Image.Image | None, str]:
    try:
        if input_image is None:
            raise RoomMorphValidationError("Upload foto ruangan sebelum generate.")

        settings = validate_settings(
            style,
            instruction,
            strength,
            steps,
            do_cfg,
            cfg_scale,
            seed,
            resolution,
        )
        prepared_image = preprocess_room_image(input_image, settings.resolution)
        prompt = build_prompt(settings.style, settings.instruction)
    except RoomMorphValidationError as exc:
        raise gr.Error(str(exc)) from exc

    if SKIP_MODEL_LOAD:
        return (
            None,
            "Inference disabled: ROOMMORPH_SKIP_MODEL_LOAD=1. "
            "No placeholder or generated image was produced.",
        )

    model_store = model_manager.MODEL_STORE
    loading_message = (
        "Loading Stable Diffusion components once; this can take a while..."
        if not model_store.loaded
        else "Model ready. Generating the room concept..."
    )
    LOGGER.info(loading_message)

    try:
        with INFERENCE_LOCK:
            models, tokenizer = model_store.load()
            _configure_pipeline_resolution(settings.resolution)
            inference_steps = int(settings.steps)
            seed = int(settings.seed)

            LOGGER.info(
                "Starting generation device=%s resolution=%s steps=%s strength=%.2f cfg=%s seed=%s",
                RUN_DEVICE,
                settings.resolution,
                inference_steps,
                settings.strength,
                settings.do_cfg,
                seed,
            )
            started = time.perf_counter()
            with torch.inference_mode():
                generated = sd_pipeline.generate(
                    prompt=prompt,
                    uncond_prompt=NEGATIVE_PROMPT,
                    input_image=prepared_image,
                    strength=settings.strength,
                    do_cfg=settings.do_cfg,
                    cfg_scale=settings.cfg_scale,
                    sampler_name="ddpm",
                    n_inference_steps=inference_steps,
                    models=models,
                    seed=seed,
                    device=RUN_DEVICE,
                    idle_device=IDLE_DEVICE,
                    tokenizer=tokenizer,
                )
            elapsed = time.perf_counter() - started

            result = to_rgb_pil_image(generated)
            output_path = save_output(result)
            LOGGER.info("Generation complete in %.2fs; saved to %s", elapsed, output_path)

        try:
            metadata = _metadata(settings, elapsed, output_path)
            if not isinstance(metadata, str):
                raise TypeError("Generation metadata must be a string.")
        except Exception:
            LOGGER.exception(
                "Output was saved but callback payload preparation failed; saved_path=%s",
                output_path,
            )
            raise

        LOGGER.info(
            "Returning Gradio output image_type=%s image_mode=%s image_size=%s saved_path=%s",
            type(result).__name__,
            result.mode,
            result.size,
            output_path,
        )
        return result, metadata
    except ModelFilesMissingError as exc:
        LOGGER.exception("Required model files are missing")
        model_store.offload()
        _clear_device_cache()
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("RoomMorph generation failed")
        model_store.offload()
        _clear_device_cache()
        if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
            message = (
                "Memori perangkat tidak cukup. Coba resolusi 256, matikan CFG, "
                "atau kurangi inference steps."
            )
        else:
            message = (
                "Generasi gagal. Periksa log server untuk detail, lalu coba lagi "
                "dengan resolusi 256 dan CFG nonaktif."
            )
        raise gr.Error(message) from exc


def generate_new_room_concept(
    style: str,
    positive_prompt: str,
    negative_prompt: str,
    resolution: int,
    steps: int,
    cfg_scale: float,
    seed: int,
) -> tuple[Image.Image | None, str]:
    try:
        settings = validate_text_to_image_settings(
            style,
            positive_prompt,
            negative_prompt,
            resolution,
            steps,
            cfg_scale,
            seed,
        )
        effective_prompt = build_text_to_image_prompt(
            settings.style, settings.positive_prompt
        )
    except RoomMorphValidationError as exc:
        raise gr.Error(str(exc)) from exc

    if SKIP_MODEL_LOAD:
        return (
            None,
            "Inference disabled: ROOMMORPH_SKIP_MODEL_LOAD=1. "
            "No placeholder or generated image was produced.",
        )

    model_store = model_manager.MODEL_STORE
    loading_message = (
        "Loading Stable Diffusion components once for text-to-image..."
        if not model_store.loaded
        else "Model ready. Generating a new room concept..."
    )
    LOGGER.info(loading_message)

    try:
        with INFERENCE_LOCK:
            models, tokenizer = model_store.load()
            _configure_pipeline_resolution(settings.resolution)
            inference_steps = int(settings.steps)
            seed = int(settings.seed)

            LOGGER.info(
                "Starting txt2img device=%s resolution=%s steps=%s cfg_scale=%.1f seed=%s",
                RUN_DEVICE,
                settings.resolution,
                inference_steps,
                settings.cfg_scale,
                seed,
            )
            started = time.perf_counter()
            with torch.inference_mode():
                generated = sd_pipeline.generate(
                    prompt=effective_prompt,
                    uncond_prompt=settings.negative_prompt,
                    input_image=None,
                    strength=1.0,
                    do_cfg=True,
                    cfg_scale=settings.cfg_scale,
                    sampler_name="ddpm",
                    n_inference_steps=inference_steps,
                    models=models,
                    seed=seed,
                    device=RUN_DEVICE,
                    idle_device=IDLE_DEVICE,
                    tokenizer=tokenizer,
                )
            elapsed = time.perf_counter() - started

            result = to_rgb_pil_image(generated)
            output_path = save_output(result, prefix="txt2img")
            LOGGER.info(
                "Text-to-image generation complete in %.2fs; saved to %s",
                elapsed,
                output_path,
            )

        try:
            metadata = _text_to_image_metadata(
                settings, effective_prompt, elapsed, output_path
            )
            if not isinstance(metadata, str):
                raise TypeError("Generation metadata must be a string.")
        except Exception:
            LOGGER.exception(
                "Text-to-image output was saved but callback payload preparation failed; saved_path=%s",
                output_path,
            )
            raise

        LOGGER.info(
            "Returning text-to-image Gradio output image_type=%s image_mode=%s image_size=%s saved_path=%s",
            type(result).__name__,
            result.mode,
            result.size,
            output_path,
        )
        return result, metadata
    except ModelFilesMissingError as exc:
        LOGGER.exception("Required model files are missing")
        model_store.offload()
        _clear_device_cache()
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("RoomMorph text-to-image generation failed")
        model_store.offload()
        _clear_device_cache()
        if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
            message = (
                "Memori perangkat tidak cukup. Coba resolusi 256 atau kurangi "
                "inference steps."
            )
        else:
            message = (
                "Generasi text-to-image gagal. Periksa log server untuk detail, "
                "lalu coba lagi dengan resolusi 256."
            )
        raise gr.Error(message) from exc
