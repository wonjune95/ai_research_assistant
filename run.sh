#!/usr/bin/env bash
# AI 리서치 어시스턴트 실행 스크립트 (macOS / Linux)
#
# 1. 가상환경(venv) 활성화 확인 및 활성화
# 2. .env 파일 존재 확인
# 3. main.py 실행

set -e
set -o pipefail

# 스크립트가 어느 위치에서 실행되든 프로젝트 루트를 기준으로 동작하도록 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="venv"

# 1. 가상환경 활성화 확인
if [ -n "${VIRTUAL_ENV:-}" ]; then
    echo "이미 활성화된 가상환경을 사용합니다: $VIRTUAL_ENV"
elif [ -f "$VENV_DIR/bin/activate" ]; then
    echo "가상환경을 활성화합니다: $VENV_DIR"
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
else
    echo "[경고] 가상환경(${VENV_DIR}/)을 찾을 수 없습니다."
    echo "  다음 명령으로 먼저 생성해주세요:"
    echo "    python -m venv ${VENV_DIR}"
    echo "    source ${VENV_DIR}/bin/activate"
    echo "    pip install -r requirements.txt"
    exit 1
fi

# 2. 환경변수(.env) 파일 존재 확인
if [ ! -f ".env" ]; then
    echo "[경고] .env 파일을 찾을 수 없습니다."
    echo "  프로젝트 루트에 .env 파일을 만들고 다음 내용을 추가해주세요:"
    echo "    OPENAI_API_KEY=sk-..."
    exit 1
fi

# 3. main.py 실행
echo "AI 리서치 어시스턴트를 시작합니다..."
python main.py
