@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
REM AI 리서치 어시스턴트 실행 스크립트 (Windows)
REM
REM 1. 가상환경(venv) 활성화 확인 및 활성화
REM 2. .env 파일 존재 확인
REM 3. main.py 실행

REM 스크립트가 어느 위치에서 실행되든 프로젝트 루트를 기준으로 동작하도록 이동
cd /d "%~dp0"

set "VENV_DIR=venv"

REM 1. 가상환경 활성화 확인
if defined VIRTUAL_ENV (
    echo 이미 활성화된 가상환경을 사용합니다: %VIRTUAL_ENV%
) else (
    if exist "%VENV_DIR%\Scripts\activate.bat" (
        echo 가상환경을 활성화합니다: %VENV_DIR%
        call "%VENV_DIR%\Scripts\activate.bat"
    ) else (
        echo [경고] 가상환경^(%VENV_DIR%\^)을 찾을 수 없습니다.
        echo   다음 명령으로 먼저 생성해주세요:
        echo     python -m venv %VENV_DIR%
        echo     %VENV_DIR%\Scripts\activate
        echo     pip install -r requirements.txt
        exit /b 1
    )
)

REM 2. 환경변수(.env) 파일 존재 확인
if not exist ".env" (
    echo [경고] .env 파일을 찾을 수 없습니다.
    echo   프로젝트 루트에 .env 파일을 만들고 다음 내용을 추가해주세요:
    echo     OPENAI_API_KEY=sk-...
    exit /b 1
)

REM 3. main.py 실행
echo AI 리서치 어시스턴트를 시작합니다...
python main.py

endlocal
