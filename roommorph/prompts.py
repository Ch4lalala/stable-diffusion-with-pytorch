from __future__ import annotations

from .exceptions import RoomMorphValidationError


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
