#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# FoodScan 3D — Local Production Launcher
# Usage:  ./start.sh          (builds frontend if needed, starts server)
#         ./start.sh --build  (force-rebuild frontend before starting)
#         ./start.sh --stop   (kill running server)
# ─────────────────────────────────────────────────────────────────
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND="$DIR/backend"
VENV="$BACKEND/venv"
DIST="$DIR/dist"
PIDFILE="$BACKEND/server.pid"
PORT="${PORT:-8000}"

# ── Stop mode ──────────────────────────────────────────────────────
if [[ "$1" == "--stop" ]]; then
  if [[ -f "$PIDFILE" ]]; then
    PID=$(cat "$PIDFILE")
    echo "Stopping FoodScan 3D (PID $PID)..."
    kill "$PID" 2>/dev/null && rm -f "$PIDFILE" && echo "Stopped."
  else
    echo "No PID file found. Killing any process on port $PORT..."
    lsof -ti:$PORT | xargs kill -9 2>/dev/null || echo "Nothing running on port $PORT."
  fi
  exit 0
fi

echo "═══════════════════════════════════════════════"
echo "  🍽  FoodScan 3D — Local Deploy"
echo "═══════════════════════════════════════════════"

# ── Check venv ─────────────────────────────────────────────────────
if [[ ! -d "$VENV" ]]; then
  echo "❌  Python venv not found at $VENV"
  echo "    Run: cd backend && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

# ── Build frontend ─────────────────────────────────────────────────
if [[ "$1" == "--build" ]] || [[ ! -f "$DIST/index.html" ]]; then
  echo "📦  Building frontend (npm run build)..."
  cd "$DIR"
  npm run build
  echo "✅  Frontend built → $DIST"
else
  echo "✅  Frontend already built (use --build to rebuild)"
fi

# ── Kill any stale server ───────────────────────────────────────────
lsof -ti:$PORT | xargs kill -9 2>/dev/null || true
sleep 1

# ── Start backend server ────────────────────────────────────────────
echo ""
echo "🚀  Starting server on http://localhost:$PORT"
echo "    Logs: $BACKEND/server.log"
echo "    Stop: ./start.sh --stop"
echo ""

cd "$BACKEND"
source "$VENV/bin/activate"
PORT=$PORT nohup python3 server.py > server.log 2>&1 &
echo $! > "$PIDFILE"
echo "Server PID $! started."
echo ""
echo "Opening browser in 40s (model loading)..."
sleep 40 && open "http://localhost:$PORT" &
