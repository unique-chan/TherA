#!/bin/bash
# Run TherA inference with weather/style palette reference caches.
#
# Usage:
#   bash scripts/infer_palette.sh /path/to/rgb [output_dir]
#
# Requires weights/reference_caches/{SUNNY,CLOUDY,RAINY,NIGHT}.pt

set -euo pipefail

cd "$(dirname "$0")/.."

RGB_DIR="${1:?RGB directory required}"
OUT_BASE="${2:-preds_palette}"

for PAL in SUNNY CLOUDY RAINY NIGHT; do
  CACHE="weights/reference_caches/${PAL}.pt"
  if [[ ! -f "$CACHE" ]]; then
    echo "Skipping ${PAL}: missing ${CACHE}"
    continue
  fi
  echo ">>> ${PAL}"
  python infer_custom.py \
    --rgb-dir "$RGB_DIR" \
    --output-dir "${OUT_BASE}/${PAL}" \
    --reference-cache "$CACHE"
done

echo ""
echo "Done. Outputs in: ${OUT_BASE}/"
