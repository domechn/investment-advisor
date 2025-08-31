#!/usr/bin/env bash
set -euo pipefail

# Usage:
# bash scripts/init_project.sh \
#   --financial-mcp-url https://github.com/<org>/financial-datasets-mcp \
#   --google-mcp-url https://github.com/<org>/Google-Search-MCP-Server

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
VENV_DIR="$ROOT_DIR/venv"
MCP_DIR="$ROOT_DIR/mcp"
VENDORS_DIR="$MCP_DIR/vendors"

FIN_URL="https://github.com/financial-datasets/mcp-server"
GOO_URL="https://github.com/mixelpixx/Google-Search-MCP-Server"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --financial-mcp-url)
      FIN_URL="$2"; shift 2;;
    --google-mcp-url)
      GOO_URL="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

echo "[1/5] Creating Python venv at $VENV_DIR"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "[2/5] Installing Python dependencies"
pip install -U pip
pip install -r "$ROOT_DIR/requirements.txt"

mkdir -p "$VENDORS_DIR"

if [[ -n "$FIN_URL" ]]; then
  echo "[3/5] Cloning Financial MCP: $FIN_URL"
  if [[ ! -d "$VENDORS_DIR/financial-datasets-mcp" ]]; then
    git clone "$FIN_URL" "$VENDORS_DIR/financial-datasets-mcp"
  else
    echo "Financial MCP already exists, skipping clone"
  fi
fi

if [[ -n "$GOO_URL" ]]; then
  echo "[4/5] Cloning Google Search MCP: $GOO_URL"
  if [[ ! -d "$VENDORS_DIR/Google-Search-MCP-Server" ]]; then
    git clone "$GOO_URL" "$VENDORS_DIR/Google-Search-MCP-Server"
  else
    echo "Google Search MCP already exists, skipping clone"
  fi
  if command -v npm >/dev/null 2>&1; then
    echo "Building Google Search MCP (npm install)"
    (cd "$VENDORS_DIR/Google-Search-MCP-Server" && npm install && npm run build || true)
  else
    echo "npm not found; skipping Google MCP build"
  fi
fi

echo "[5/5] Writing mcp.json"
LOCAL_CFG="$MCP_DIR/mcp.json"
cat > "$LOCAL_CFG" <<JSON
{
  "financial": {
    "transport": "stdio",
    "command": "uv",
    "args": [
      "--directory",
      "$VENDORS_DIR/financial-datasets-mcp",
      "run",
      "server.py"
    ],
    "env": {
      "FINANCIAL_DATASETS_API_KEY": "\${FINANCIAL_DATASETS_API_KEY}"
    }
  },
  "google-search": {
    "transport": "stdio",
    "command": "node",
    "args": [
      "$VENDORS_DIR/Google-Search-MCP-Server/dist/google-search.js"
    ],
    "env": {
      "GOOGLE_API_KEY": "\${GOOGLE_API_KEY}",
      "GOOGLE_SEARCH_ENGINE_ID": "\${GOOGLE_SEARCH_ENGINE_ID}"
    }
  }
}
JSON

echo "Done. Activate venv with: source $VENV_DIR/bin/activate"
echo "Optional: export OPENAI_API_KEY and MCP keys (FINANCIAL_DATASETS_API_KEY, GOOGLE_API_KEY, GOOGLE_SEARCH_ENGINE_ID)"


