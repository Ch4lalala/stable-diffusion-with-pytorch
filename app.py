from __future__ import annotations

import gc
import inspect
import logging
import math
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import CLIPTokenizer


ROOT_DIR = Path(__file__).resolve().parent
SD_DIR = ROOT_DIR / "sd"
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"

# The educational SD modules intentionally use flat imports for the notebook.
if str(SD_DIR) not in sys.path:
    sys.path.insert(0, str(SD_DIR))

import model_loader  # noqa: E402
import pipeline as sd_pipeline  # noqa: E402


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

STYLE_PROMPTS = {
    "Japandi": "Japandi interior design, warm neutral colors, natural wood furniture, calm minimal decor",
    "Minimalist": "minimalist interior design, uncluttered surfaces, clean lines, functional furniture",
    "Scandinavian": "Scandinavian interior design, light wood, soft neutral textiles, bright airy atmosphere",
    "Industrial": "industrial interior design, exposed materials, dark metal accents, practical open character",
    "Modern": "modern interior design, refined clean lines, balanced materials, contemporary furniture",
    "Cozy": "cozy interior design, warm layered lighting, comfortable textiles, inviting lived-in details",
}

NEGATIVE_PROMPT = (
    "distorted room, warped walls, duplicate furniture, floating objects, "
    "deformed architecture, blurry, low quality"
)


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


class RoomMorphValidationError(ValueError):
    pass


class ModelFilesMissingError(FileNotFoundError):
    pass


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


def _missing_model_files() -> list[Path]:
    return [path for path in (MODEL_FILE, VOCAB_FILE, MERGES_FILE) if not path.is_file()]


def _missing_files_message(missing: list[Path]) -> str:
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
                missing = _missing_model_files()
                if missing:
                    raise ModelFilesMissingError(_missing_files_message(missing))

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
                LOGGER.info("Stable Diffusion components loaded on idle device %s", IDLE_DEVICE)

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
INFERENCE_LOCK = threading.Lock()


def _clear_device_cache() -> None:
    if RUN_DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    elif RUN_DEVICE.type == "mps":
        mps_module = getattr(torch, "mps", None)
        if mps_module is not None and hasattr(mps_module, "empty_cache"):
            mps_module.empty_cache()
    gc.collect()


def preprocess_room_image(
    source: Image.Image | np.ndarray | str | Path,
    resolution: int,
) -> Image.Image:
    if resolution not in SUPPORTED_RESOLUTIONS or resolution % 8 != 0:
        raise RoomMorphValidationError("Resolusi harus 256 atau 512 dan habis dibagi 8.")

    try:
        if isinstance(source, Image.Image):
            image = source.copy()
        elif isinstance(source, np.ndarray):
            image = Image.fromarray(source)
        elif isinstance(source, (str, Path)):
            with Image.open(source) as opened:
                image = opened.copy()
        else:
            raise RoomMorphValidationError("Format gambar tidak dikenali.")

        image = ImageOps.exif_transpose(image)
        has_transparency = "A" in image.getbands() or "transparency" in image.info
        if has_transparency:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (240, 240, 240, 255))
            background.alpha_composite(rgba)
            image = background.convert("RGB")
        else:
            image = image.convert("RGB")

        return ImageOps.fit(
            image,
            (resolution, resolution),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
    except RoomMorphValidationError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise RoomMorphValidationError(
            "Gambar tidak dapat dibaca. Gunakan file PNG, JPG, atau JPEG yang valid."
        ) from exc


def build_prompt(style: str, instruction: str = "") -> str:
    if style not in STYLE_PROMPTS:
        raise RoomMorphValidationError("Pilih salah satu style interior yang tersedia.")

    parts = [
        "a realistic interior photograph of the same room",
        STYLE_PROMPTS[style],
    ]
    cleaned_instruction = " ".join(instruction.split())
    if cleaned_instruction:
        parts.append(cleaned_instruction)
    parts.extend(
        [
            "preserve the room layout and architectural structure",
            "soft natural lighting",
            "highly detailed",
        ]
    )
    return ", ".join(parts)


def build_text_to_image_prompt(style: str, positive_prompt: str) -> str:
    if style not in STYLE_PROMPTS:
        raise RoomMorphValidationError("Pilih salah satu style interior yang tersedia.")

    cleaned_prompt = " ".join(positive_prompt.split())
    if not cleaned_prompt:
        raise RoomMorphValidationError("Positive prompt tidak boleh kosong.")

    return ", ".join(
        [
            "a realistic interior photograph",
            STYLE_PROMPTS[style],
            cleaned_prompt,
            "cohesive room layout",
            "soft natural lighting",
            "highly detailed",
        ]
    )


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


def update_cfg_scale_interactivity(enabled: bool) -> dict[str, Any]:
    return gr.update(interactive=bool(enabled))


def _save_output(image: Image.Image, prefix: str = "roommorph") -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = OUTPUT_DIR / f"{prefix}_{timestamp}_{uuid.uuid4().hex[:8]}.png"
    image.save(output_path, format="PNG")
    return output_path


def _to_rgb_pil_image(generated: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(generated, Image.Image):
        image = generated.copy()
    else:
        image = Image.fromarray(np.asarray(generated))

    image = image.convert("RGB")
    image.load()
    if image.mode != "RGB" or image.width <= 0 or image.height <= 0:
        raise ValueError("Generated output is not a valid RGB image.")
    return image


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


def _initial_status() -> str:
    if SKIP_MODEL_LOAD:
        return (
            "Smoke-test mode is active (ROOMMORPH_SKIP_MODEL_LOAD=1). "
            "The interface works, but inference is disabled and no AI image will be returned."
        )

    missing = _missing_model_files()
    if missing:
        return _missing_files_message(missing)

    return (
        "Model files found. The model is loaded once on the first generation, "
        "so the first request has additional startup time."
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

    loading_message = (
        "Loading Stable Diffusion components once; this can take a while..."
        if not MODEL_STORE.loaded
        else "Model ready. Generating the room concept..."
    )
    LOGGER.info(loading_message)

    try:
        with INFERENCE_LOCK:
            models, tokenizer = MODEL_STORE.load()
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

            result = _to_rgb_pil_image(generated)
            output_path = _save_output(result)
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
        MODEL_STORE.offload()
        _clear_device_cache()
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("RoomMorph generation failed")
        MODEL_STORE.offload()
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

    loading_message = (
        "Loading Stable Diffusion components once for text-to-image..."
        if not MODEL_STORE.loaded
        else "Model ready. Generating a new room concept..."
    )
    LOGGER.info(loading_message)

    try:
        with INFERENCE_LOCK:
            models, tokenizer = MODEL_STORE.load()
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

            result = _to_rgb_pil_image(generated)
            output_path = _save_output(result, prefix="txt2img")
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
        MODEL_STORE.offload()
        _clear_device_cache()
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("RoomMorph text-to-image generation failed")
        MODEL_STORE.offload()
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


APP_CSS = """
.gradio-container {
    max-width: none !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
}
#app-shell {
    box-sizing: border-box;
    max-width: 1180px !important;
    width: calc(100% - 32px) !important;
    margin-left: auto !important;
    margin-right: auto !important;
}
#app-header, #app-header * { text-align: center; }
#runtime-status { padding: 10px 12px; border-left: 3px solid var(--primary-500); }
#settings-panel { width: 100%; }
#generate-button { min-height: 44px; }
#main-tabs {
    width: 100%;
    min-width: 0 !important;
    max-width: 100%;
}
#main-tabs > .tab-wrapper {
    width: 100%;
    min-width: 0;
    max-width: 100%;
    height: auto;
    min-height: 44px;
}
#main-tabs > .tab-wrapper > .tab-container[role="tablist"] {
    display: grid !important;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0 !important;
    width: 100%;
    min-width: 0;
    max-width: 100%;
    height: auto;
    min-height: 44px;
}
#main-tabs > .tab-wrapper > .tab-container[role="tablist"] > button[role="tab"] {
    box-sizing: border-box;
    width: 100% !important;
    min-width: 0 !important;
    height: auto;
    min-height: 44px;
    padding: 8px 12px;
    justify-content: center;
    text-align: center;
    white-space: normal;
    line-height: 1.25;
}
#main-tabs > .tab-wrapper > .tab-container.visually-hidden > button {
    width: 50% !important;
    min-width: 0 !important;
    max-width: 50% !important;
    padding: 0 !important;
    overflow: hidden;
}
#main-tabs > .tab-wrapper > .overflow-menu { display: none !important; }
@media (max-width: 480px) {
    #main-tabs > .tab-wrapper > .tab-container[role="tablist"] > button[role="tab"] {
        padding: 8px 6px;
    }
}
"""

GRADIO_LAUNCH_ACCEPTS_CSS = "css" in inspect.signature(gr.Blocks.launch).parameters


def create_demo() -> gr.Blocks:
    blocks_kwargs: dict[str, Any] = {
        "title": "RoomMorph AI",
        "analytics_enabled": False,
    }
    if not GRADIO_LAUNCH_ACCEPTS_CSS:
        blocks_kwargs["css"] = APP_CSS

    with gr.Blocks(**blocks_kwargs) as demo:
        with gr.Column(elem_id="app-shell", min_width=0):
            with gr.Column(elem_id="app-header", min_width=0):
                gr.Markdown(
                    "# RoomMorph AI\n"
                    "Redesign an existing room or generate a new interior concept using "
                    "this repository's custom Stable Diffusion 1.5 pipeline."
                )
                gr.Markdown(
                    f"**Runtime device:** `{RUN_DEVICE.type}`  \n"
                    "Free CPU inference can take several minutes. MPS or CUDA is recommended "
                    "for local use."
                )
            gr.Markdown(_initial_status(), elem_id="runtime-status")

            with gr.Tabs(elem_id="main-tabs"):
                with gr.Tab("Redesign Existing Room"):
                    with gr.Row(equal_height=True):
                        input_image = gr.Image(
                            label="Original room",
                            sources=["upload"],
                            type="pil",
                            image_mode=None,
                            format="png",
                            height=420,
                        )
                        generated_image = gr.Image(
                            label="Generated design",
                            type="pil",
                            image_mode="RGB",
                            format="png",
                            interactive=False,
                            height=420,
                        )

                    with gr.Group(elem_id="settings-panel"):
                        gr.Markdown("### Generation settings")

                        with gr.Row(equal_height=True):
                            style = gr.Dropdown(
                                choices=list(STYLE_PROMPTS),
                                value="Japandi",
                                label="Interior style",
                            )
                            resolution = gr.Radio(
                                choices=[("256 x 256", 256), ("512 x 512", 512)],
                                value=256,
                                label="Resolution",
                            )

                        instruction = gr.Textbox(
                            label="Additional design instruction",
                            placeholder="Example: keep the windows and add more storage",
                            lines=3,
                            max_lines=5,
                        )

                        with gr.Row(equal_height=True):
                            strength = gr.Number(
                                value=0.60,
                                minimum=MIN_STRENGTH,
                                maximum=MAX_STRENGTH,
                                step=0.05,
                                precision=2,
                                label="Strength",
                                info="How far the design can move from the original room.",
                            )
                            steps = gr.Number(
                                value=10,
                                minimum=MIN_STEPS,
                                maximum=MAX_STEPS,
                                step=1,
                                precision=0,
                                label="Inference steps",
                                info="More steps can add detail but take longer.",
                            )

                        with gr.Row(equal_height=True):
                            do_cfg = gr.Checkbox(
                                value=False,
                                label="Enable classifier-free guidance (CFG)",
                                info="Usually improves prompt adherence but increases computation.",
                            )
                            cfg_scale = gr.Number(
                                value=7.5,
                                minimum=MIN_CFG_SCALE,
                                maximum=MAX_CFG_SCALE,
                                step=0.5,
                                precision=1,
                                label="CFG scale (used only when CFG is enabled)",
                                info="Prompt guidance strength; available only when CFG is enabled.",
                                interactive=False,
                            )
                            seed = gr.Number(
                                value=42,
                                minimum=0,
                                maximum=MAX_SEED,
                                step=1,
                                precision=0,
                                label="Seed",
                                info="Repeat a result with the same settings.",
                            )

                    generate_button = gr.Button(
                        "Generate Design", variant="primary", elem_id="generate-button"
                    )
                    generation_status = gr.Textbox(
                        value=_initial_status(),
                        label="Generation status and metadata",
                        lines=11,
                        interactive=False,
                    )

                    do_cfg.change(
                        fn=update_cfg_scale_interactivity,
                        inputs=do_cfg,
                        outputs=cfg_scale,
                        queue=False,
                        show_progress="hidden",
                    )

                    generate_button.click(
                        fn=generate_design,
                        inputs=[
                            input_image,
                            style,
                            instruction,
                            strength,
                            steps,
                            do_cfg,
                            cfg_scale,
                            seed,
                            resolution,
                        ],
                        outputs=[generated_image, generation_status],
                        concurrency_limit=1,
                        concurrency_id="roommorph-inference",
                        show_progress="full",
                    )

                with gr.Tab("Generate New Room Concept"):
                    with gr.Row(equal_height=True):
                        with gr.Column(min_width=280):
                            txt2img_style = gr.Dropdown(
                                choices=list(STYLE_PROMPTS),
                                value="Japandi",
                                label="Room style preset",
                            )
                            txt2img_positive_prompt = gr.Textbox(
                                label="Positive prompt",
                                placeholder="Example: a sunlit open-plan living room with built-in shelves",
                                lines=5,
                                max_lines=7,
                            )
                            txt2img_negative_prompt = gr.Textbox(
                                value=NEGATIVE_PROMPT,
                                label="Negative prompt",
                                lines=4,
                                max_lines=6,
                            )

                        with gr.Column(min_width=280):
                            txt2img_generated_image = gr.Image(
                                label="Generated room concept",
                                type="pil",
                                image_mode="RGB",
                                format="png",
                                interactive=False,
                                height=420,
                            )

                    with gr.Group(elem_id="txt2img-settings-panel"):
                        gr.Markdown("### Generation settings")
                        with gr.Row(equal_height=True):
                            txt2img_resolution = gr.Radio(
                                choices=[("256 x 256", 256), ("512 x 512", 512)],
                                value=256,
                                label="Resolution",
                            )
                            txt2img_steps = gr.Number(
                                value=10,
                                minimum=MIN_STEPS,
                                maximum=MAX_STEPS,
                                step=1,
                                precision=0,
                                label="Inference steps",
                                info="More steps can add detail but take longer.",
                            )
                            txt2img_cfg_scale = gr.Number(
                                value=7.5,
                                minimum=MIN_CFG_SCALE,
                                maximum=MAX_CFG_SCALE,
                                step=0.5,
                                precision=1,
                                label="CFG scale",
                                info="Higher values follow the prompt more strongly.",
                            )
                            txt2img_seed = gr.Number(
                                value=42,
                                minimum=0,
                                maximum=MAX_SEED,
                                step=1,
                                precision=0,
                                label="Seed",
                                info="Repeat a result with the same settings.",
                            )

                    txt2img_generate_button = gr.Button(
                        "Generate Room Concept",
                        variant="primary",
                        elem_id="txt2img-generate-button",
                    )
                    txt2img_generation_status = gr.Textbox(
                        value=_initial_status(),
                        label="Text-to-image generation metadata",
                        lines=14,
                        interactive=False,
                    )

                    txt2img_generate_button.click(
                        fn=generate_new_room_concept,
                        inputs=[
                            txt2img_style,
                            txt2img_positive_prompt,
                            txt2img_negative_prompt,
                            txt2img_resolution,
                            txt2img_steps,
                            txt2img_cfg_scale,
                            txt2img_seed,
                        ],
                        outputs=[
                            txt2img_generated_image,
                            txt2img_generation_status,
                        ],
                        concurrency_limit=1,
                        concurrency_id="roommorph-inference",
                        show_progress="full",
                    )

    return demo.queue(max_size=8, default_concurrency_limit=1, api_open=False)


demo = create_demo()


if __name__ == "__main__":
    launch_kwargs = {"css": APP_CSS} if GRADIO_LAUNCH_ACCEPTS_CSS else {}
    demo.launch(**launch_kwargs)
