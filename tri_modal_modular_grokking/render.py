from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from modular_addition.data import number_to_words


@dataclass(frozen=True)
class RenderConfig:
    height: int = 64
    width: int = 128
    channels: int = 1
    render_style: str = "digit"
    font_size: int = 36
    normalize: bool = True
    antialias: bool = True
    margin: int = 4


def font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def display_text(value: int, render_style: str) -> str:
    if render_style == "digit":
        return str(value)
    if render_style == "words":
        return number_to_words(value)
    if render_style == "digit_or_words":
        return str(value) if value < 100 else number_to_words(value)
    raise ValueError(f"unknown render_style: {render_style}")


def render_text(text: str, cfg: RenderConfig) -> Image.Image:
    scale = 2 if cfg.antialias else 1
    image = Image.new("L", (cfg.width * scale, cfg.height * scale), color=255)
    draw = ImageDraw.Draw(image)
    used_font = font(cfg.font_size * scale)
    bbox = draw.textbbox((0, 0), text, font=used_font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = max(cfg.margin * scale, (image.width - text_width) // 2)
    y = max(cfg.margin * scale, (image.height - text_height) // 2 - bbox[1])
    draw.text((x, y), text, fill=0, font=used_font)
    if cfg.antialias:
        image = image.resize((cfg.width, cfg.height), Image.Resampling.LANCZOS)
    return image


def image_to_tensor(image: Image.Image, *, normalize: bool = True) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32)
    if normalize:
        array = array / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def render_operand_image(value: int, cfg: RenderConfig) -> torch.Tensor:
    return image_to_tensor(render_text(display_text(value, cfg.render_style), cfg), normalize=cfg.normalize)


def render_answer_image(value: int, cfg: RenderConfig) -> torch.Tensor:
    return render_operand_image(value, cfg)


def render_equation_scene(a: int, b: int, modulus: int, answer: int | None, cfg: RenderConfig) -> dict[str, torch.Tensor]:
    suffix = "?" if answer is None else str(answer)
    text = f"({a} + {b}) mod {modulus} = {suffix}"
    return {"image": image_to_tensor(render_text(text, cfg), normalize=cfg.normalize)}


def save_tensor_image(tensor: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = tensor.detach().cpu().squeeze(0).clamp(0, 1).mul(255).byte().numpy()
    Image.fromarray(array, mode="L").save(path)

