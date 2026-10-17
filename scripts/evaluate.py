"""
Model Evaluation and Diagnostic Audit Script
============================================

Evaluates a trained or prototype Diabetic Retinopathy model checkpoint against
validation/test sets. Computes:
1. Multi-class Accuracy and Quadratic Weighted Kappa (QWK).
2. Per-grade Precision, Recall, and F1-score.
3. Referable Diabetic Retinopathy (RDR, Grade >= 2) Clinical Sensitivity and Specificity.
4. Confusion Matrix and classification summary reports.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
import torch
import yaml

# Allow script execution from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from oculi.data.dataset import create_dataloaders
from oculi.models.backbone import build_model
from oculi.models.predictor import DR_CLASSES

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("oculi.evaluate")


def run_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Using compute device: {device}")

    # Locate checkpoint
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        checkpoint_path = os.path.join(
            config.get("training", {}).get("checkpoint_dir", "checkpoints"),
            config.get("training", {}).get("checkpoint_name", "best_model.pth"),
        )

    has_checkpoint = checkpoint_path and os.path.exists(checkpoint_path)
    if has_checkpoint:
        logger.info(f"Evaluating checkpoint: {checkpoint_path}")
        model = build_model(
            num_classes=5,
            pretrained=False,
            checkpoint_path=checkpoint_path,
            device=device,
        )
    else:
        logger.warning(
            f"Checkpoint not found at '{checkpoint_path}'. "
            "Evaluating baseline uncalibrated prototype architecture."
        )
        model = build_model(num_classes=5, pretrained=True, device=device)

    model.eval()

    # Data loading
    data_cfg = config.get("data", {})
    image_size = data_cfg.get("image_size", 224)
    test_csv = args.data_csv or data_cfg.get("test_csv") or data_cfg.get("val_csv")
    image_dir = args.image_dir or data_cfg.get("image_dir")

    is_dry_run = args.dry_run
    dummy_samples = args.synthetic_samples if is_dry_run else 0

    if not is_dry_run and not (test_csv and Path(test_csv).exists()):
        logger.warning(
            f"Test data CSV not found at '{test_csv}'. Running evaluation with {args.synthetic_samples} synthetic samples."
        )
        is_dry_run = True
        dummy_samples = args.synthetic_samples

    _, eval_loader = create_dataloaders(
        val_csv=test_csv if not is_dry_run else None,
        image_dir=image_dir if not is_dry_run else None,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=0,
        dummy_samples=dummy_samples,
    )

    all_preds = []
    all_targets = []
    all_probs = []

    logger.info(f"Running inference on {len(eval_loader.dataset)} evaluation images...")

    with torch.no_grad():
        for batch in eval_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    # Multi-class Metrics
    accuracy = float(np.mean(y_true == y_pred))
    try:
        qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    except Exception:
        qwk = 0.0

    conf_mat = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    target_names = [DR_CLASSES[i] for i in range(5)]
    clf_report = classification_report(
        y_true,
        y_pred,
        labels=list(range(5)),
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    # Referable Diabetic Retinopathy (RDR) Clinical Metrics (Grade >= 2)
    # Binary classification: 0 (Non-referable: Grade 0,1) vs 1 (Referable: Grade 2,3,4)
    rdr_true = (y_true >= 2).astype(int)
    rdr_pred = (y_pred >= 2).astype(int)

    tp = int(np.sum((rdr_true == 1) & (rdr_pred == 1)))
    tn = int(np.sum((rdr_true == 0) & (rdr_pred == 0)))
    fp = int(np.sum((rdr_true == 0) & (rdr_pred == 1)))
    fn = int(np.sum((rdr_true == 1) & (rdr_pred == 0)))

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0  # Recall of RDR
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0  # TNR of Non-RDR
    rdr_acc = float((tp + tn) / max(1, len(rdr_true)))

    results = {
        "overall_accuracy": round(accuracy, 4),
        "quadratic_weighted_kappa": round(qwk, 4),
        "referable_dr_metrics": {
            "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "accuracy": round(rdr_acc, 4),
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
        },
        "per_class_metrics": clf_report,
        "confusion_matrix": conf_mat.tolist(),
        "is_checkpoint_loaded": has_checkpoint,
        "checkpoint_path": checkpoint_path if has_checkpoint else "None (ImageNet Prototype)",
    }

    # Print Report
    print("\n" + "=" * 70)
    print("           OCULI: DIABETIC RETINOPATHY EVALUATION AUDIT")
    print("=" * 70)
    print(f"Model Source           : {results['checkpoint_path']}")
    print(f"Total Evaluated Images : {len(y_true)}")
    print(f"Overall Accuracy       : {accuracy * 100:.2f}%")
    print(f"Quadratic Kappa (QWK)  : {qwk:.4f}")
    print("-" * 70)
    print("Referable DR (Grade >= 2) Clinical Screening Triage:")
    print(f"  - Sensitivity (Recall) : {sensitivity * 100:.2f}% (Target: >85% for clinical utility)")
    print(f"  - Specificity          : {specificity * 100:.2f}% (Target: >85% for clinical utility)")
    print(f"  - Binary Accuracy      : {rdr_acc * 100:.2f}%")
    print("-" * 70)
    print("Confusion Matrix (Rows: Ground Truth, Columns: Predictions):")
    print("      " + "  ".join([f"G{i}" for i in range(5)]))
    for i, row in enumerate(conf_mat):
        row_str = "  ".join([f"{val:3d}" for val in row])
        print(f"G{i} : [{row_str}]  ({DR_CLASSES[i]})")
    print("=" * 70 + "\n")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Evaluation report written to: {out_path}")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Oculi DR Screening Model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pth checkpoint")
    parser.add_argument("--data-csv", type=str, default=None, help="Path to test CSV")
    parser.add_argument("--image-dir", type=str, default=None, help="Path to test retinal images")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('cpu' or 'cuda')")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate on synthetic sample dataset")
    parser.add_argument("--synthetic-samples", type=int, default=20, help="Number of synthetic samples")
    parser.add_argument("--output-json", type=str, default=None, help="Save evaluation metrics to JSON file")
    return parser.parse_args()


if __name__ == "__main__":
    run_evaluation(parse_args())
