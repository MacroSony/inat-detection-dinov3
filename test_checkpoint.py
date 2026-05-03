#!/usr/bin/env python3
"""Run a trained RF-DETR/DINOv3 checkpoint and save annotated images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
RFDETR_SRC = ROOT / "rfdetr" / "src"
if str(RFDETR_SRC) not in sys.path:
    sys.path.insert(0, str(RFDETR_SRC))

from custom_backbone import apply_monkey_patch  # noqa: E402
from rfdetr.detr import RFDETR  # noqa: E402


DEFAULT_CHECKPOINT = ROOT / "lightning_logs" / "version_2" / "checkpoints" / "epoch=49-step=450000.ckpt"
DEFAULT_CONFIG = ROOT / "dinov3_config.yaml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the trained DINOv3 RF-DETR checkpoint and save annotated outputs.",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Image file or directory containing images to run inference on.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Checkpoint to load. Default: {DEFAULT_CHECKPOINT}",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Training config YAML. Default: {DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "test_outputs",
        help="Directory where annotated images are written. Default: ./test_outputs",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection confidence threshold. Default: 0.5",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use, e.g. cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for images when input is a directory.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> tuple[dict, list[str]]:
    with config_path.open("r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    model_config = dict(raw_config["model"]["model_config"])
    class_names = list(raw_config["model"]["train_config"].get("class_names", []))
    return model_config, class_names


def find_images(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    pattern = "**/*" if recursive else "*"
    images = sorted(
        path
        for path in input_path.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No images found in {input_path}")
    return images


def color_for_class(class_id: int | None) -> tuple[int, int, int]:
    palette = [
        (230, 57, 70),
        (29, 53, 87),
        (42, 157, 143),
        (244, 162, 97),
        (131, 56, 236),
        (255, 183, 3),
        (0, 150, 199),
        (108, 117, 125),
        (214, 40, 40),
    ]
    if class_id is None:
        return (230, 57, 70)
    return palette[int(class_id) % len(palette)]


def draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    x, y = xy
    label_box = draw.textbbox((x, y), text, font=font)
    label_w = label_box[2] - label_box[0]
    label_h = label_box[3] - label_box[1]
    label_y = max(0, y - label_h - 6)
    draw.rectangle((x, label_y, x + label_w + 8, label_y + label_h + 6), fill=color)
    draw.text((x + 4, label_y + 3), text, fill=(255, 255, 255), font=font)


def annotate_image(image_path: Path, detections, output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    class_names = detections.data.get("class_name", [])
    class_ids = detections.class_id if detections.class_id is not None else []
    confidences = detections.confidence if detections.confidence is not None else []

    for idx, box in enumerate(detections.xyxy):
        class_id = int(class_ids[idx]) if len(class_ids) > idx else None
        class_name = str(class_names[idx]) if len(class_names) > idx else str(class_id)
        confidence = float(confidences[idx]) if len(confidences) > idx else 0.0
        color = color_for_class(class_id)
        x1, y1, x2, y2 = [float(v) for v in box]

        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        draw_label(draw, (x1, y1), f"{class_name} {confidence:.2f}", color, font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def output_path_for(image_path: Path, input_root: Path, output_dir: Path) -> Path:
    if input_root.is_dir():
        rel_path = image_path.relative_to(input_root)
    else:
        rel_path = image_path.name
    rel_path = Path(rel_path)
    return output_dir / rel_path.with_name(f"{rel_path.stem}_annotated{rel_path.suffix}")


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    config_path = args.config.resolve()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    images = find_images(input_path, args.recursive)

    apply_monkey_patch()
    model_config, class_names = load_config(config_path)
    model_config["pretrain_weights"] = str(checkpoint)
    model_config["device"] = args.device

    model = RFDETR(**model_config)
    if class_names:
        model.model.class_names = class_names

    print(f"Loaded checkpoint: {checkpoint}")
    print(f"Running {len(images)} image(s) on {args.device} with threshold={args.threshold}")

    for image_path in images:
        input_image = Image.open(image_path).convert("RGB")
        detections = model.predict(input_image, threshold=args.threshold)
        output_path = output_path_for(image_path, input_path, output_dir)
        annotate_image(image_path, detections, output_path)
        print(f"{image_path} -> {output_path} ({len(detections)} detections)")


if __name__ == "__main__":
    main()
