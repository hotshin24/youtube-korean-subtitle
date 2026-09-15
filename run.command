#!/bin/zsh
set -euo pipefail

cd "${0:A:h}"

SUPPORT_DIR="$PWD/.runtime-local"
APP_VENV="$SUPPORT_DIR/venv"
RUNTIME_DIR="$SUPPORT_DIR/runtime"
PID_FILE="$RUNTIME_DIR/server.pid"
LOG_FILE="$RUNTIME_DIR/server.log"
PYTHON_BIN="/opt/homebrew/bin/python3.12"
APP_URL="http://localhost:8501"

function pause_on_error() {
  local exit_code=$?
  if [[ $exit_code -ne 0 ]]; then
    echo
    echo "실행 중 오류가 발생했습니다 (코드: $exit_code)."
    [[ -f "$LOG_FILE" ]] && echo "로그: $LOG_FILE"
    read "?이 창을 닫으려면 Enter 키를 누르세요."
  fi
}
trap pause_on_error EXIT

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

if /usr/bin/curl --silent --fail --max-time 1 "$APP_URL/_stcore/health" >/dev/null 2>&1; then
  /usr/bin/open "$APP_URL"
  exit 0
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "FFmpeg가 설치되어 있지 않습니다. 먼저 'brew install ffmpeg'를 실행해 주세요."
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python 3.12가 필요합니다. 먼저 'brew install python@3.12'를 실행해 주세요."
  exit 1
fi

mkdir -p "$SUPPORT_DIR" "$RUNTIME_DIR"

if [[ ! -x "$APP_VENV/bin/python" ]]; then
  echo "처음 실행을 준비합니다. 패키지 설치에 몇 분 걸릴 수 있습니다…"
  "$PYTHON_BIN" -m venv "$APP_VENV"
  "$APP_VENV/bin/python" -m pip install --disable-pip-version-check --upgrade pip
  "$APP_VENV/bin/python" -m pip install --disable-pip-version-check -r requirements.txt
fi

echo "앱 서버를 시작합니다…"
: > "$LOG_FILE"
nohup "$APP_VENV/bin/python" -m streamlit run app.py \
  --server.headless=true --server.port=8501 \
  >"$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"

for _ in {1..60}; do
  if /usr/bin/curl --silent --fail --max-time 1 "$APP_URL/_stcore/health" >/dev/null 2>&1; then
    /usr/bin/open "$APP_URL"
    echo "브라우저를 열었습니다."
    exit 0
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    /usr/bin/open -a TextEdit "$LOG_FILE"
    exit 1
  fi
  sleep 1
done

/usr/bin/open -a TextEdit "$LOG_FILE"
exit 1
