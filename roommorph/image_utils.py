from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .config import OUTPUT_DIR, SUPPORTED_RESOLUTIONS
from .exceptions import RoomMorphValidationError


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


def save_output(image: Image.Image, prefix: str = "roommorph") -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = OUTPUT_DIR / f"{prefix}_{timestamp}_{uuid.uuid4().hex[:8]}.png"
    image.save(output_path, format="PNG")
    return output_path


def to_rgb_pil_image(generated: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(generated, Image.Image):
        image = generated.copy()
    else:
        image = Image.fromarray(np.asarray(generated))

    image = image.convert("RGB")
    image.load()
    if image.mode != "RGB" or image.width <= 0 or image.height <= 0:
        raise ValueError("Generated output is not a valid RGB image.")
    return image
