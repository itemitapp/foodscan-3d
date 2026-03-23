#!/bin/bash
# FoodScan 3D — Start Local Python Backend
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv
source venv/bin/activate

# Install any missing extras
pip install scikit-learn -q 2>/dev/null || true

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   FoodScan 3D — MapAnything Local Backend v2     ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "  🌐 API:   http://localhost:8000"
echo "  📖 Docs:  http://localhost:8000/docs"
echo ""
echo "  First run downloads ~5 GB model weights"
echo "  Subsequent runs load from cache in seconds"
echo ""
echo "  Press Ctrl+C to stop"
echo ""

uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
