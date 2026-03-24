#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# FoodScan 3D — Development Mode
# Starts both the Vite dev server (port 5173) and the Python backend
# (port 8000) with hot-reload on the frontend.
# ─────────────────────────────────────────────────────────────────
DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND="$DIR/backend"
VENV="$BACKEND/venv"

if [[ ! -d "$VENV" ]]; then
  echo "❌  Python venv not found. Run setup first."
  exit 1
fi

# Kill any stale processes
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:5173 | xargs kill -9 2>/dev/null || true
sleep 1

echo "🔧  FoodScan 3D — Dev Mode"
echo "    Frontend: http://localhost:5173 (hot reload)"
echo "    Backend:  http://localhost:8000"
echo "    Press Ctrl+C to stop both."
echo ""

# Start backend in background
cd "$BACKEND"
source "$VENV/bin/activate"
python3 server.py > server.log 2>&1 &
BACKEND_PID=$!
echo "  Backend PID: $BACKEND_PID   (logs: backend/server.log)"

# Start frontend dev server in foreground
cd "$DIR"
npm run dev

# When Ctrl+C hits npm, also kill backend
kill $BACKEND_PID 2>/dev/null
echo "Stopped."
