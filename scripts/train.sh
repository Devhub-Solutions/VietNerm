#!/bin/bash
# ============================================================================
# VietNerm - Train NER Model
# ============================================================================
# Usage:
#   ./scripts/train.sh cccd
#   ./scripts/train.sh giay_ra_vien
#   ./scripts/train.sh cccd --epochs 10 --batch_size 32
# ============================================================================

set -euo pipefail

DOC_TYPE=${1:-cccd}
shift 2>/dev/null || true  # Remove doc_type from args, pass rest through

echo "======================================"
echo "VietNerm - Model Training"
echo "======================================"
echo "Document type : ${DOC_TYPE}"
echo "Extra args    : $*"
echo "======================================"

# Ensure we're in the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

# Run training
python training/train.py --doc "${DOC_TYPE}" "$@"

echo ""
echo "[DONE] Model trained for ${DOC_TYPE}"
echo "  Output: models/phobert/${DOC_TYPE}/"
