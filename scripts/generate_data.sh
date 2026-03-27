#!/bin/bash
# ============================================================================
# VietNerm - Generate Synthetic Dataset
# ============================================================================
# Usage:
#   ./scripts/generate_data.sh cccd 50000
#   ./scripts/generate_data.sh giay_ra_vien 10000
#   ./scripts/generate_data.sh              # defaults: cccd, 10000 samples
# ============================================================================

set -euo pipefail

DOC_TYPE=${1:-cccd}
SIZE=${2:-10000}

echo "======================================"
echo "VietNerm - Synthetic Data Generation"
echo "======================================"
echo "Document type : ${DOC_TYPE}"
echo "Sample size   : ${SIZE}"
echo "======================================"

# Ensure we're in the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

# Run dataset generation
python synthetic/generate_dataset.py --doc "${DOC_TYPE}" --size "${SIZE}"

echo ""
echo "[DONE] Dataset generated for ${DOC_TYPE} (${SIZE} samples)"
echo "  Raw:  datasets/raw/${DOC_TYPE}/"
echo "  NER:  datasets/ner/${DOC_TYPE}/"
