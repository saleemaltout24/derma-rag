#!/usr/bin/env bash
# Run from project root after ISIC_2019_Training_Input.zip has finished downloading.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/isic2019/raw"
ZIP="$RAW/ISIC_2019_Training_Input.zip"
EXPECTED_BYTES=9771110400  # ~9.1 GB per ISIC challenge page

cd "$ROOT"
source venv/bin/activate

if [[ ! -f "$ZIP" ]]; then
  echo "Missing: $ZIP"
  echo "Wait for the download to finish (see data/isic2019/raw/isic_download.log)."
  exit 1
fi

SIZE=$(stat -f%z "$ZIP" 2>/dev/null || stat -c%s "$ZIP")
if (( SIZE < EXPECTED_BYTES - 50000000 )); then
  echo "Zip still downloading: $(du -h "$ZIP" | cut -f1) (need ~9.1 GB)"
  echo "Check: tail -f $RAW/isic_download.log"
  exit 1
fi

echo "== Unzip training images (may take several minutes) =="
unzip -q -o "$ZIP" -d "$RAW"

echo "== Build train/val folders =="
python scripts/prepare_isic2019.py

echo "== Train classifier v3 (this takes a long time on CPU) =="
python scripts/train_skin_classifier.py

echo "== Eval on data/eval/ =="
python scripts/eval_classifier.py

echo ""
echo "Done. Restart backend: uvicorn app:app --host 127.0.0.1 --port 8000 --reload"
echo "Then test your AK photos in the browser."
