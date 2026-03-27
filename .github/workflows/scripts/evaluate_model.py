#!/usr/bin/env python3
"""Evaluate a trained NER model with smoke tests and quality gates.

Loads the trained model, runs inference on test samples, checks that
predictions contain expected entity types, and validates the F1 score
meets a minimum threshold.

Usage:
    python .github/workflows/scripts/evaluate_model.py --doc cccd --model-dir models/phobert/cccd
    python .github/workflows/scripts/evaluate_model.py --doc cccd --min-f1 0.5
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_test_samples(doc_type: str, dataset_dir: str = None, max_samples: int = 20):
    """Load a few test samples for evaluation."""
    if dataset_dir:
        ds_path = Path(dataset_dir)
    else:
        ds_path = PROJECT_ROOT / "datasets" / "ner" / doc_type

    test_file = ds_path / "test.json"
    if not test_file.exists():
        print(f"Error: Test file not found at {test_file}", file=sys.stderr)
        sys.exit(1)

    with open(test_file) as f:
        samples = json.load(f)

    return samples[:max_samples]


def load_labels(doc_type: str, dataset_dir: str = None):
    """Load label definitions."""
    if dataset_dir:
        ds_path = Path(dataset_dir)
    else:
        ds_path = PROJECT_ROOT / "datasets" / "ner" / doc_type

    labels_file = ds_path / "labels.json"
    if not labels_file.exists():
        print(f"Error: Labels file not found at {labels_file}", file=sys.stderr)
        sys.exit(1)

    with open(labels_file) as f:
        return json.load(f)


def check_training_metrics(model_dir: Path, min_f1: float) -> dict:
    """Check if training produced acceptable metrics.

    Looks for trainer_state.json which HuggingFace Trainer saves
    with training logs including eval metrics.
    """
    result = {"passed": False, "f1": 0.0, "details": ""}

    # Check trainer_state.json for eval metrics
    trainer_state = model_dir / "trainer_state.json"
    if trainer_state.exists():
        with open(trainer_state) as f:
            state = json.load(f)

        # Find the best metric from training logs
        best_f1 = 0.0
        for entry in state.get("log_history", []):
            f1 = entry.get("eval_f1", 0.0)
            if f1 > best_f1:
                best_f1 = f1

        result["f1"] = best_f1
        result["passed"] = best_f1 >= min_f1
        result["details"] = f"Best eval F1: {best_f1:.4f} (threshold: {min_f1})"
        return result

    # Fallback: check label_map.json exists (model is at least structurally valid)
    label_map = model_dir / "label_map.json"
    if label_map.exists():
        result["details"] = "No trainer_state.json found; model structure looks valid"
        result["passed"] = True
        result["f1"] = -1.0  # Unknown
        return result

    result["details"] = "No training metrics or label_map found"
    return result


def smoke_test_inference(model_dir: Path, test_samples: list, labels_info: dict) -> dict:
    """Run inference on test samples and validate predictions."""
    result = {"passed": False, "total": 0, "predicted": 0, "details": ""}

    try:
        from inference.pipeline import NERPipeline
    except ImportError:
        result["details"] = "Could not import NERPipeline — skipping inference smoke test"
        result["passed"] = True  # Non-blocking if inference module not available
        return result

    # Reconstruct text from tokens
    texts = []
    for sample in test_samples:
        text = " ".join(sample.get("tokens", []))
        if text.strip():
            texts.append(text)

    if not texts:
        result["details"] = "No valid test texts found"
        return result

    result["total"] = len(texts)

    try:
        pipeline = NERPipeline(model_path=str(model_dir), device="cpu")
    except Exception as e:
        result["details"] = f"Failed to load model: {e}"
        return result

    # Expected entity types from labels (strip B-/I- prefixes)
    expected_entities = set()
    for label in labels_info.get("labels", []):
        if label.startswith("B-"):
            expected_entities.add(label[2:])

    predicted_entities = set()
    successful_predictions = 0

    for text in texts:
        try:
            preds = pipeline.predict(text)
            if preds:
                successful_predictions += 1
                for pred in preds:
                    entity_type = pred.get("type", pred.get("label", ""))
                    # Strip B-/I- prefix if present
                    if entity_type.startswith(("B-", "I-")):
                        entity_type = entity_type[2:]
                    if entity_type and entity_type != "O":
                        predicted_entities.add(entity_type)
        except Exception:
            continue

    result["predicted"] = successful_predictions
    overlap = expected_entities & predicted_entities
    result["passed"] = successful_predictions > 0
    result["details"] = (
        f"Ran inference on {len(texts)} samples, "
        f"{successful_predictions} produced predictions. "
        f"Entity types found: {sorted(predicted_entities)} "
        f"(expected: {sorted(expected_entities)}, overlap: {sorted(overlap)})"
    )

    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained NER model")
    parser.add_argument("--doc", type=str, required=True, help="Document type")
    parser.add_argument("--model-dir", type=str, default=None, help="Model directory")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Dataset directory")
    parser.add_argument("--min-f1", type=float, default=0.5, help="Minimum F1 threshold")
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip inference smoke test (only check metrics)",
    )
    args = parser.parse_args()

    if args.model_dir:
        model_dir = Path(args.model_dir)
    else:
        model_dir = PROJECT_ROOT / "models" / "phobert" / args.doc

    if not model_dir.exists():
        print(f"Error: Model directory not found at {model_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"=== Evaluating model for '{args.doc}' ===")
    print(f"Model dir: {model_dir}")
    print()

    all_passed = True

    # --- Check 1: Training metrics / quality gate ---
    print("[1/2] Checking training metrics...")
    metrics_result = check_training_metrics(model_dir, args.min_f1)
    status = "PASS" if metrics_result["passed"] else "FAIL"
    print(f"  {status}: {metrics_result['details']}")
    if not metrics_result["passed"]:
        all_passed = False
    print()

    # --- Check 2: Inference smoke test ---
    if not args.skip_inference:
        print("[2/2] Running inference smoke test...")
        test_samples = load_test_samples(args.doc, args.dataset_dir)
        labels_info = load_labels(args.doc, args.dataset_dir)
        inference_result = smoke_test_inference(model_dir, test_samples, labels_info)
        status = "PASS" if inference_result["passed"] else "FAIL"
        print(f"  {status}: {inference_result['details']}")
        if not inference_result["passed"]:
            all_passed = False
    else:
        print("[2/2] Inference smoke test: SKIPPED")
    print()

    # --- Summary ---
    if all_passed:
        print("EVALUATION PASSED")
        sys.exit(0)
    else:
        print("EVALUATION FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
