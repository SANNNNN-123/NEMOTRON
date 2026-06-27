#!/usr/bin/env bash
# Assemble a minimal static site for visualization.html (Vercel / local preview).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SITE="$ROOT/site"
DATA="$SITE/data"

python3 "$ROOT/scripts/build_viz_manifest.py" --repo-root "$ROOT"

rm -rf "$SITE"
mkdir -p "$DATA"

cp "$ROOT/visualization.html" "$SITE/index.html"
cp "$ROOT/visualization.html" "$SITE/visualization.html"

for f in \
  viz_manifest.json \
  problems.jsonl \
  train.csv \
  predictions_90.7.csv \
  predictions.csv \
  predictions_2.csv
do
  src="$ROOT/data/$f"
  if [[ ! -f "$src" ]]; then
    echo "Missing required data file: $src" >&2
    exit 1
  fi
  cp "$src" "$DATA/$f"
done

echo "Site ready at $SITE ($(du -sh "$SITE" | cut -f1))"
