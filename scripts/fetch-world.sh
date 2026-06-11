#!/bin/bash
# Poll a Spaitial world request until complete, download splat + simplified mesh.
# Usage: fetch-world.sh <request_id> <level_id>
set -euo pipefail
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
source "$PROJECT/.env"
REQ_ID="$1"
LEVEL_ID="$2"
LEVEL_DIR="$PROJECT/public/levels/$LEVEL_ID"
mkdir -p "$LEVEL_DIR"
API="https://api.spaitial.ai"
AUTH=(-H "Authorization: Bearer $SPAITIAL_API_KEY")

echo "[$LEVEL_ID] polling world generation ($REQ_ID)..."
while true; do
  STATUS=$(curl -s "${AUTH[@]}" "$API/v1/worlds/requests/$REQ_ID/status" | /usr/bin/python3 -c "import sys,json;print(json.load(sys.stdin).get('status','?'))")
  echo "[$LEVEL_ID] world status: $STATUS ($(date +%H:%M:%S))"
  case "$STATUS" in
    COMPLETED) break ;;
    FAILED|CANCELLED) echo "[$LEVEL_ID] GENERATION $STATUS" >&2; exit 1 ;;
  esac
  sleep 20
done

echo "[$LEVEL_ID] downloading splat..."
curl -sSL "${AUTH[@]}" "$API/v1/worlds/requests/$REQ_ID/splat" -o "$LEVEL_DIR/world.spz"
ls -la "$LEVEL_DIR/world.spz"

echo "[$LEVEL_ID] starting mesh-simplified export..."
curl -sS -X POST "${AUTH[@]}" "$API/v1/worlds/requests/$REQ_ID/exports/mesh-simplified" >/dev/null || true

while true; do
  EXPORT=$(curl -s "${AUTH[@]}" "$API/v1/worlds/requests/$REQ_ID/exports/mesh-simplified")
  ESTATUS=$(echo "$EXPORT" | /usr/bin/python3 -c "import sys,json;print(json.load(sys.stdin).get('status','?'))")
  echo "[$LEVEL_ID] export status: $ESTATUS ($(date +%H:%M:%S))"
  case "$ESTATUS" in
    READY)
      URL=$(echo "$EXPORT" | /usr/bin/python3 -c "import sys,json;print(json.load(sys.stdin)['download_url'])")
      curl -sSL "${AUTH[@]}" "$URL" -o "$LEVEL_DIR/mesh_simplified.ply"
      ls -la "$LEVEL_DIR/mesh_simplified.ply"
      break ;;
    FAILED) echo "[$LEVEL_ID] EXPORT FAILED" >&2; exit 1 ;;
  esac
  sleep 20
done
echo "[$LEVEL_ID] DONE"
