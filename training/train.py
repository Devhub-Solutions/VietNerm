"""Universal NER training CLI.

Usage:
    python training/train.py --doc cccd --epochs 10
    python training/train.py --doc giay_ra_vien --epochs 7 --batch_size 16
    python training/train.py --doc vehicle_registration
"""

import argparse
import os
import sys
from pathlib import Path

import yaml

# Ensure CUDA_LAUNCH_BLOCKING is set before any torch import
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PhoBERT NER model for Vietnamese documents"
    )
    parser.add_argument(
        "--doc",
        type=str,
        required=True,
        help="Document type (e.g., cccd, giay_ra_vien, vehicle_registration)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override num_train_epochs from config",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override per_device_train_batch_size from config",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override learning rate from config",
    )
    parser.add_argument(
        "--real_data",
        type=str,
        default=None,
        help="Path to real annotation JSON file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Override config file path (default: training/config/{doc}.yaml)",
    )
    args = parser.parse_args()

    # Load config
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = PROJECT_ROOT / "training" / "config" / f"{args.doc}.yaml"
        if not config_path.exists():
            config_path = PROJECT_ROOT / "training" / "config" / "default.yaml"
            print(f"[INFO] No config for '{args.doc}', using default.yaml")

    print(f"==> Loading config from {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides
    if args.epochs is not None:
        config["num_train_epochs"] = args.epochs
    if args.batch_size is not None:
        config["per_device_train_batch_size"] = args.batch_size
        config["per_device_eval_batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate

    # Check dataset exists
    dataset_dir = PROJECT_ROOT / "datasets" / "ner" / args.doc
    if not dataset_dir.exists():
        print(f"Error: Dataset not found at {dataset_dir}")
        print(f"Run: python synthetic/generate_dataset.py --doc {args.doc} --size 2000")
        sys.exit(1)

    # Output directory
    output_dir = PROJECT_ROOT / "models" / "phobert" / args.doc

    # Real data path
    real_data_path = Path(args.real_data) if args.real_data else None

    # Train
    from training.trainer import PhoBERTNERTrainer

    trainer = PhoBERTNERTrainer(config)
    trainer.train(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        real_data_path=real_data_path,
    )

    print(f"\n==> Model saved to {output_dir}")
    print("==> Training complete!")


if __name__ == "__main__":
    main()
