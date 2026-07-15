# AI 리서치 어시스턴트

OpenAI Chat Completions API를 활용한 **대화형 커맨드라인 리서치 어시스턴트**입니다.
전문 리서치 어시스턴트 페르소나를 가진 AI와 대화하며 조사·분석 작업을 진행하고,
대화 내용을 요약하거나 파일로 저장할 수 있습니다.

> 1주차 바이브 코딩 실습으로 시작된 프로젝트입니다. 현재는 콘솔 기반 대화 루프와
> 기본적인 상태 관리·재시도·저장 기능까지 구현되어 있습니다.

## 주요 기능

- **전문 리서치 어시스턴트 페르소나** — `config/prompts.py`에 정의된 시스템 메시지를 통해
  사용자 의도 파악, 명확화 질문, 출처 언급 습관을 갖춘 어조로 응답합니다.
- **대화 히스토리 관리** — `ConversationManager`가 system/user/assistant 메시지를
  누적 관리하며, 완료된 대화 턴 수(`turn_count`)를 추적합니다.
- **대화 상태 관리** — 사용자 입력에 "조사", "분석", "리서치", "알아봐", "찾아봐" 등의
  키워드가 포함되면 `researching` 상태로, 그 외에는 `responding` 상태로 전환됩니다.
  (현재는 키워드 기반 규칙이며, 4주차에 LLM 기반 의도 분류로 고도화할 예정입니다.)
- **API 호출 자동 재시도** — OpenAI API 호출이 일시적으로 실패하면 지수 백오프
  (2초 → 4초 → 8초)로 최대 3회까지 재시도합니다. 인증 오류(API 키 문제)는
  재시도해도 해결되지 않으므로 즉시 명확한 안내 메시지를 반환합니다.
- **대화 요약** — `summary` 명령으로 지금까지의 대화를 3문장으로 요약합니다.
  대화가 충분히 길지 않으면 안내 메시지를 반환합니다.
- **대화 저장** — `save` 명령 또는 종료 시 확인을 통해 대화 히스토리를
  `data/` 디렉터리에 타임스탬프가 포함된 JSON 파일로 저장합니다.
- **명령어 기반 인터페이스** — `save` / `summary` / `quit`·`exit`·`종료` 명령어와
  구분선을 활용한 가독성 높은 콘솔 UI를 제공합니다.
- **중앙화된 설정** — 모델명, temperature, 최대 토큰 수, 재시도 횟수, 저장 경로 등을
  `config/settings.py`에서 한 곳에서 관리합니다.
- **구체적인 에러 처리** — API 키 누락/오류, 파일 저장 실패 등을 커스텀 예외
  (`src/exceptions.py`)로 구분해 사용자 친화적인 메시지와 로그를 제공합니다.
- **로깅** — `logging` 모듈을 통해 DEBUG/INFO/WARNING/ERROR 레벨로 동작을 기록합니다.

## 요구사항

- Python 3.10 이상
- [OpenAI API 키](https://platform.openai.com/api-keys)
- 주요 패키지 (`requirements.txt` 참고)

  | 패키지 | 버전 |
  |---|---|
  | `openai` | 2.45.0 |
  | `python-dotenv` | 1.0.0 |

## 설치 방법

### 1. 가상환경 생성 및 활성화

**Windows (PowerShell)**

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python -m venv venv
source venv/bin/activate
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

### 3. 환경변수 설정

프로젝트 루트에 `.env` 파일을 만들고 OpenAI API 키를 입력합니다. 이 파일은
`.gitignore`에 포함되어 있어 git에 커밋되지 않습니다.

```
OPENAI_API_KEY=sk-...
```

`OPENAI_API_KEY`가 설정되어 있지 않으면 실행 시 `ConfigurationError`와 함께
명확한 안내 메시지가 출력됩니다.

## 사용 방법

### 기본 실행

```bash
python main.py
```

실행하면 환영 메시지와 사용 가능한 명령어 안내가 표시된 뒤, `You:` 프롬프트로
자유롭게 질문을 입력할 수 있습니다.

```
============================================================
AI 리서치 어시스턴트에 오신 것을 환영합니다!
============================================================
궁금한 내용을 자유롭게 입력해주세요.

사용 가능한 명령어
  save     : 지금까지의 대화를 파일로 저장합니다.
  summary  : 지금까지의 대화를 3문장으로 요약합니다.
  quit / exit / 종료 : 프로그램을 종료합니다.
============================================================

You: 최근 생성형 AI 시장 동향을 조사해줘
============================================================
AI: (리서치 모드로 응답이 생성됩니다)
============================================================
```

### 명령어 설명

| 명령어 | 설명 |
|---|---|
| (일반 입력) | AI와 자유롭게 대화합니다. "조사/분석/리서치/알아봐/찾아봐"가 포함되면 리서치 모드로 처리됩니다. |
| `save` | 지금까지의 대화 히스토리를 `data/conversation_YYYYMMDD_HHMMSS.json`으로 저장합니다. |
| `summary` | 지금까지의 대화를 3문장으로 요약해 출력합니다. |
| `quit` / `exit` / `종료` | 프로그램을 종료합니다. 종료 전 저장 여부(`y`/`n`)를 확인합니다. |

Ctrl+C로도 언제든 대화를 중단할 수 있습니다.

## 프로젝트 구조

```
ai_reseach_assistant/
├── config/
│   ├── prompts.py            # 시스템 프롬프트 (RESEARCH_ASSISTANT_SYSTEM_MESSAGE)
│   └── settings.py           # 모델명/temperature/재시도 횟수 등 전역 설정값
├── data/                     # 대화 저장 파일(JSON)이 생성되는 디렉터리
├── src/
│   ├── __init__.py
│   ├── conversation_manager.py   # 대화 히스토리 관리, API 호출, 재시도, 상태 판단, 요약
│   └── exceptions.py         # 커스텀 예외 (ConfigurationError, ConversationSaveError 등)
├── .env                      # OPENAI_API_KEY 등 환경변수 (git에 커밋하지 않음)
├── .gitignore
├── CLAUDE.md                 # 프로젝트 진행 상황 및 작업 지침
├── main.py                   # 실행 진입점 (CLI 루프, 명령어 처리)
├── README.md
└── requirements.txt
```

## 향후 계획

- **2~3주차**: 웹 검색 등 외부 소스 연동, 조사 결과를 구조화된 리포트(Markdown 등)로
  `data/`에 저장하는 기능 추가
- **4주차**: `ConversationManager.determine_state`의 키워드 기반 판단을 LLM 기반
  의도 분류로 고도화 (코드에 이미 `NOTE` 주석으로 명시됨)
- **5주차**: 리서치 파이프라인 고도화 및 사용성 개선 (예: 대화 이어하기, 검색 결과
  출처 정리, 설정 옵션 확장 등)

> 위 계획은 `CLAUDE.md`의 진행 상황 체크리스트를 바탕으로 한 잠정적인 방향이며,
> 실습이 진행됨에 따라 구체화됩니다.

## 라이선스

이 프로젝트는 [MIT License](https://opensource.org/licenses/MIT)를 따릅니다.
