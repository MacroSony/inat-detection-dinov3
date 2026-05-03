#!/usr/bin/env python3
"""Evaluate the trained RF-DETR/DINOv3 checkpoint on a COCO validation split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


ROOT = Path(__file__).resolve().parent
RFDETR_SRC = ROOT / "rfdetr" / "src"
if str(RFDETR_SRC) not in sys.path:
    sys.path.insert(0, str(RFDETR_SRC))

from custom_backbone import apply_monkey_patch  # noqa: E402
from rfdetr.detr import RFDETR  # noqa: E402


DEFAULT_CHECKPOINT = ROOT / "lightning_logs" / "version_2" / "checkpoints" / "epoch=49-step=450000.ckpt"
DEFAULT_CONFIG = ROOT / "dinov3_config.yaml"
DEFAULT_ANN = ROOT / "rf_dataset" / "valid" / "_annotations.coco.json"
DEFAULT_IMAGE_ROOT = ROOT / "data" / "images"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the COCO validation set.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANN)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "validation_eval")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--predict-threshold",
        type=float,
        default=0.001,
        help="Low confidence cutoff used while collecting predictions for AP/AR.",
    )
    parser.add_argument(
        "--metric-threshold",
        type=float,
        default=0.5,
        help="Confidence cutoff used for precision/recall/F1/accuracy reporting.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold used for thresholded TP/FP/FN matching.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N images.")
    parser.add_argument("--skip-coco", action="store_true", help="Skip COCO AP/AR evaluation.")
    return parser.parse_args()


def load_model(config_path: Path, checkpoint_path: Path, device: str) -> RFDETR:
    with config_path.open("r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    model_config = dict(raw_config["model"]["model_config"])
    class_names = list(raw_config["model"]["train_config"].get("class_names", []))
    model_config["pretrain_weights"] = str(checkpoint_path)
    model_config["device"] = device

    apply_monkey_patch()
    model = RFDETR(**model_config)
    if class_names:
        model.model.class_names = class_names
    return model


def xywh_to_xyxy(box: list[float]) -> np.ndarray:
    x, y, w, h = box
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def load_ground_truth(annotation_path: Path, image_root: Path, limit: int | None) -> tuple[list[dict], dict, dict, dict]:
    with annotation_path.open("r", encoding="utf-8") as f:
        coco_data = json.load(f)

    images = list(coco_data["images"])
    if limit is not None:
        images = images[:limit]
    image_ids = {img["id"] for img in images}

    categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
    gt_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco_data["annotations"]:
        if ann["image_id"] not in image_ids or ann.get("iscrowd", 0):
            continue
        gt_by_image[ann["image_id"]].append(
            {
                "bbox_xyxy": xywh_to_xyxy(ann["bbox"]),
                "category_id": int(ann["category_id"]),
            }
        )

    missing = [img["file_name"] for img in images if not (image_root / img["file_name"]).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} validation images under {image_root}; first: {missing[0]}")

    return images, gt_by_image, categories, coco_data


def detections_to_records(image_id: int, detections) -> list[dict[str, Any]]:
    records = []
    class_ids = detections.class_id if detections.class_id is not None else []
    scores = detections.confidence if detections.confidence is not None else []

    for idx, box in enumerate(detections.xyxy):
        x1, y1, x2, y2 = [float(v) for v in box]
        records.append(
            {
                "image_id": int(image_id),
                "category_id": int(class_ids[idx]),
                "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                "score": float(scores[idx]),
            }
        )
    return records


def match_class_aware(preds: list[dict], gts: list[dict], conf_thr: float, iou_thr: float) -> tuple[list[dict], int]:
    kept = sorted((p for p in preds if p["score"] >= conf_thr), key=lambda p: p["score"], reverse=True)
    matched_gt: set[int] = set()
    matches = []

    for pred in kept:
        pred_box = xywh_to_xyxy(pred["bbox"])
        best_iou = 0.0
        best_idx = None
        for idx, gt in enumerate(gts):
            if idx in matched_gt or pred["category_id"] != gt["category_id"]:
                continue
            iou = float(box_iou_xyxy(pred_box[None, :], gt["bbox_xyxy"][None, :])[0, 0])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_idx is not None and best_iou >= iou_thr:
            matched_gt.add(best_idx)
            matches.append({"pred": pred, "gt": gts[best_idx], "iou": best_iou, "tp": True})
        else:
            matches.append({"pred": pred, "gt": None, "iou": 0.0, "tp": False})

    return matches, len(gts) - len(matched_gt)


def match_class_agnostic(preds: list[dict], gts: list[dict], conf_thr: float, iou_thr: float) -> tuple[int, int]:
    kept = sorted((p for p in preds if p["score"] >= conf_thr), key=lambda p: p["score"], reverse=True)
    matched_gt: set[int] = set()
    localized = 0
    localized_correct_class = 0

    for pred in kept:
        pred_box = xywh_to_xyxy(pred["bbox"])
        best_iou = 0.0
        best_idx = None
        for idx, gt in enumerate(gts):
            if idx in matched_gt:
                continue
            iou = float(box_iou_xyxy(pred_box[None, :], gt["bbox_xyxy"][None, :])[0, 0])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx is not None and best_iou >= iou_thr:
            matched_gt.add(best_idx)
            localized += 1
            if pred["category_id"] == gts[best_idx]["category_id"]:
                localized_correct_class += 1

    return localized, localized_correct_class


def compute_threshold_metrics(
    predictions_by_image: dict[int, list[dict]],
    gt_by_image: dict[int, list[dict]],
    categories: dict[int, str],
    conf_thr: float,
    iou_thr: float,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    totals = defaultdict(float)
    per_class = {
        cid: defaultdict(float, {"class_id": cid, "class_name": name})
        for cid, name in categories.items()
    }
    matched_ious: list[float] = []

    for image_id, gts in gt_by_image.items():
        preds = predictions_by_image.get(image_id, [])
        matches, fn = match_class_aware(preds, gts, conf_thr, iou_thr)
        localized, localized_correct = match_class_agnostic(preds, gts, conf_thr, iou_thr)

        totals["fn"] += fn
        totals["localized"] += localized
        totals["localized_correct_class"] += localized_correct

        gt_counts = defaultdict(int)
        for gt in gts:
            gt_counts[gt["category_id"]] += 1
        for cid, count in gt_counts.items():
            per_class[cid]["gt"] += count

        for match in matches:
            pred = match["pred"]
            cid = pred["category_id"]
            if match["tp"]:
                totals["tp"] += 1
                per_class[cid]["tp"] += 1
                matched_ious.append(match["iou"])
            else:
                totals["fp"] += 1
                per_class[cid]["fp"] += 1

        matched_by_class = defaultdict(int)
        for match in matches:
            if match["tp"]:
                matched_by_class[match["gt"]["category_id"]] += 1
        for cid, gt_count in gt_counts.items():
            per_class[cid]["fn"] += gt_count - matched_by_class[cid]

    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    detection_accuracy = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    classification_accuracy = (
        totals["localized_correct_class"] / totals["localized"] if totals["localized"] else 0.0
    )

    rows = []
    for cid in sorted(per_class):
        row = dict(per_class[cid])
        row.setdefault("gt", 0.0)
        row.setdefault("tp", 0.0)
        row.setdefault("fp", 0.0)
        row.setdefault("fn", 0.0)
        ctp, cfp, cfn = row["tp"], row["fp"], row["fn"]
        row["precision"] = ctp / (ctp + cfp) if ctp + cfp else 0.0
        row["recall"] = ctp / (ctp + cfn) if ctp + cfn else 0.0
        row["f1"] = (
            2 * row["precision"] * row["recall"] / (row["precision"] + row["recall"])
            if row["precision"] + row["recall"]
            else 0.0
        )
        row["accuracy"] = ctp / (ctp + cfp + cfn) if ctp + cfp + cfn else 0.0
        rows.append(row)

    overall = {
        "confidence_threshold": conf_thr,
        "iou_threshold": iou_thr,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy_tp_over_tp_fp_fn": detection_accuracy,
        "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "classification_accuracy_on_localized_boxes": classification_accuracy,
        "localized_box_matches": int(totals["localized"]),
    }
    return overall, rows


def run_coco_eval(annotation_path: Path, predictions: list[dict], image_ids: list[int]) -> dict[str, float]:
    if not predictions:
        return {}

    coco_gt = COCO(str(annotation_path))
    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = image_ids
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    stats = evaluator.stats
    return {
        "mAP_50_95": float(stats[0]),
        "mAP_50": float(stats[1]),
        "mAP_75": float(stats[2]),
        "mAP_small": float(stats[3]),
        "mAP_medium": float(stats[4]),
        "mAP_large": float(stats[5]),
        "mAR_1": float(stats[6]),
        "mAR_10": float(stats[7]),
        "mAR_100": float(stats[8]),
        "mAR_small": float(stats[9]),
        "mAR_medium": float(stats[10]),
        "mAR_large": float(stats[11]),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    config_path = args.config.resolve()
    annotation_path = args.annotations.resolve()
    image_root = args.image_root.resolve()
    output_dir = args.output_dir.resolve()

    for path, label in [(checkpoint, "checkpoint"), (config_path, "config"), (annotation_path, "annotations")]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")

    images, gt_by_image, categories, _ = load_ground_truth(annotation_path, image_root, args.limit)
    image_ids = [int(img["id"]) for img in images]

    model = load_model(config_path, checkpoint, args.device)

    predictions = []
    predictions_by_image: dict[int, list[dict]] = {}
    total = len(images)
    for start in range(0, total, args.batch_size):
        batch = images[start : start + args.batch_size]
        pil_images = [Image.open(image_root / img["file_name"]).convert("RGB") for img in batch]
        detections = model.predict(pil_images, threshold=args.predict_threshold, include_source_image=False)
        if not isinstance(detections, list):
            detections = [detections]

        for img, det in zip(batch, detections):
            records = detections_to_records(int(img["id"]), det)
            predictions.extend(records)
            predictions_by_image[int(img["id"])] = records

        print(f"Processed {min(start + len(batch), total)}/{total} images", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "predictions.coco.json"
    with pred_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f)

    threshold_metrics, per_class_rows = compute_threshold_metrics(
        predictions_by_image,
        gt_by_image,
        categories,
        args.metric_threshold,
        args.iou_threshold,
    )
    coco_metrics = {} if args.skip_coco else run_coco_eval(annotation_path, predictions, image_ids)
    report = {
        "checkpoint": str(checkpoint),
        "annotations": str(annotation_path),
        "image_root": str(image_root),
        "num_images": len(images),
        "num_ground_truth_boxes": int(sum(len(v) for v in gt_by_image.values())),
        "num_predictions_at_predict_threshold": len(predictions),
        "threshold_metrics": threshold_metrics,
        "coco_metrics": coco_metrics,
        "per_class": per_class_rows,
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    write_csv(output_dir / "per_class_metrics.csv", per_class_rows)

    print("\nThreshold metrics")
    for key, value in threshold_metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    if coco_metrics:
        print("\nCOCO metrics")
        for key, value in coco_metrics.items():
            print(f"{key}: {value:.4f}")

    insect = next((r for r in per_class_rows if r["class_name"] == "Insecta"), None)
    if insect:
        print("\nInsecta")
        for key in ["gt", "tp", "fp", "fn", "precision", "recall", "f1", "accuracy"]:
            value = insect[key]
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    print(f"\nWrote {output_dir / 'metrics.json'}")
    print(f"Wrote {output_dir / 'per_class_metrics.csv'}")
    print(f"Wrote {pred_path}")


if __name__ == "__main__":
    main()
