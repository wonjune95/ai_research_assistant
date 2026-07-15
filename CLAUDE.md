# AI 리서치 어시스턴트

**1주차 바이브 코딩 실습 - AI 리서치 어시스턴트** 프로젝트입니다.

## 개요

OpenAI Chat Completions API를 사용해 대화형으로 리서치를 돕는 커맨드라인 도구를 만드는 실습입니다.

## 구조

- `main.py` — 실행 진입점, CLI 루프 (save/summary/종료 명령어 포함)
- `src/conversation_manager.py` — 대화 기록(history) 관리, OpenAI API 호출/재시도,
  상태 판단, 요약, 저장된 대화 불러오기(`load_conversation`)
- `src/exceptions.py` — 커스텀 예외 (`ConfigurationError`, `ConversationSaveError`,
  `ConversationLoadError` 등)
- `config/prompts.py` — 시스템 프롬프트(페르소나) 정의
- `config/settings.py` — 모델명/temperature/재시도 횟수 등 전역 설정값
- `data/` — 대화 저장 파일 및 데모용 샘플 데이터
  (`data/sample_conversations.json` — 시나리오 4종)
- `tests/` — pytest 단위 테스트 (`test_conversation_manager.py`)
- `run.sh`, `run.bat` — 가상환경 활성화 확인 → `.env` 확인 → `main.py` 실행 스크립트
- `.env` — `OPENAI_API_KEY` 보관 (git에 커밋 금지)

## 진행 상황

**1주차 실습 완료** (2026-07-15 기준)

- [x] 프로젝트 스캐폴딩 생성 (폴더 구조, requirements.txt, .gitignore, README)
- [x] `ConversationManager` 기본 구현 (대화 기록 유지, OpenAI 호출)
- [x] `main.py` CLI 루프 구현 (save/summary/종료 명령어 포함)
- [x] 시스템 프롬프트(페르소나) 작성 — `config/prompts.py`
- [x] 대화 상태 관리 (idle/responding/researching, 키워드 기반 감지)
- [x] API 호출 재시도(지수 백오프) 및 커스텀 예외 기반 에러 처리
- [x] 대화 요약 기능 (`summarize_conversation`)
- [x] 대화 저장/불러오기 기능 (`save_conversation`, `load_conversation`)
- [x] 설정값 중앙화 (`config/settings.py`)
- [x] 단위 테스트 작성 (`tests/`, pytest 18개 통과)
- [x] 실행 스크립트 작성 (`run.sh`, `run.bat`)
- [x] 코드 리뷰 및 리팩터링 (중복 제거, 함수 복잡도 완화, PEP 8 정리)
- [x] 데모용 샘플 대화 데이터 (`data/sample_conversations.json`)
- [x] API 키 설정 및 `ConversationManager` 초기화 동작 확인
      (실제 `.env` 키로 save/summary/종료 명령어 흐름을 CLI에서 직접 실행해 검증함)
- [ ] `chat()`을 통한 실제 OpenAI 응답 왕복 테스트
      (API 크레딧이 소모되는 실제 대화 호출은 이번 세션에서 아직 수행하지 않음)
- [ ] 웹 검색 등 외부 소스 연동 (2~3주차 예정 — README "향후 계획" 참고)

## 참고

- 상위 디렉토리(`ai_reseach_assistant/`) 루트에 별도의 `.env` 파일이 존재하며 실제 API 키가 들어있음. 이 프로젝트의 `.env`와는 별개이므로 혼동 주의.
