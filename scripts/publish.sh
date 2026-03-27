#!/bin/bash
# ============================================================================
# VietNerm - Publish Model to HuggingFace
# ============================================================================
# Usage:
#   ./scripts/publish.sh cccd username
#   ./scripts/publish.sh giay_ra_vien vietnerm
#   ./scripts/publish.sh cccd username --private
# ============================================================================

set -euo pipefail

DOC_TYPE=${1:-cccd}
USERNAME=${2:-vietnerm}
shift 2 2>/dev/null || true  # Remove doc_type and username, pass rest through

REPO="${USERNAME}/phobert-${DOC_TYPE}-ner"

echo "======================================"
echo "VietNerm - Publish to HuggingFace"
echo "======================================"
echo "Document type : ${DOC_TYPE}"
echo "Repository    : ${REPO}"
echo "Extra args    : $*"
echo "======================================"

# Ensure we're in the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

# Publish model
echo ""
echo "[1/2] Publishing model..."
python huggingface/push_model.py --doc "${DOC_TYPE}" --repo "${REPO}" "$@"

# Publish dataset
echo ""
echo "[2/2] Publishing dataset..."
DATASET_REPO="${USERNAME}/vietnerm-${DOC_TYPE}-dataset"
python huggingface/push_dataset.py --doc "${DOC_TYPE}" --repo "${DATASET_REPO}" "$@"

echo ""
echo "[DONE] Published to HuggingFace"
echo "  Model:   https://huggingface.co/${REPO}"
echo "  Dataset: https://huggingface.co/datasets/${DATASET_REPO}"
