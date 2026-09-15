#!/bin/zsh
set -euo pipefail

cd "${0:A:h}"

APP_VENV="/Users/shz/Documents/Codex/.venvs/youtube-korean-script"
PYTHON_BIN="/opt/homebrew/bin/python3.12"

function pause_on_error() {
  local exit_code=$?
  if [[ $exit_code -ne 0 ]]; then
    echo
    echo "실행 중 오류가 발생했습니다 (코드: $exit_code)."
    read "?이 창을 닫으려면 Enter 키를 누르세요."
  fi
}
trap pause_on_error EXIT

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "FFmpeg가 설치되어 있지 않습니다. 먼저 'brew install ffmpeg'를 실행해 주세요."
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python 3.12가 필요합니다. 먼저 'brew install python@3.12'를 실행해 주세요."
  exit 1
fi

if [[ ! -x "$APP_VENV/bin/python" ]]; then
  echo "처음 실행을 준비합니다…"
  mkdir -p "${APP_VENV:h}"
  "$PYTHON_BIN" -m venv "$APP_VENV"
  "$APP_VENV/bin/python" -m pip install --upgrade pip
  "$APP_VENV/bin/python" -m pip install -r requirements.txt
fi

echo "앱을 백그라운드에서 시작합니다…"
zsh ./start-background.sh
