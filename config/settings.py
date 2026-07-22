"""AI 리서치 어시스턴트 전역 설정값.

src/conversation_manager.py, main.py 등 여러 모듈에서 공통으로 쓰이던
하드코딩된 값들을 한 곳에 모아 관리한다. 값을 바꿔야 할 때 이 파일만
수정하면 되도록 하기 위함이다.
"""

# OpenAI Chat Completions API 호출 시 사용할 기본 모델명
DEFAULT_MODEL: str = "gpt-4o-mini"

# 일반 대화 응답 생성 시 사용할 기본 temperature (창의성 정도, 0~2)
DEFAULT_TEMPERATURE: float = 0.7

# 응답 생성 시 허용할 최대 토큰 수
MAX_TOKENS: int = 1000

# API 호출 실패 시 재시도할 최대 횟수 (지수 백오프 적용)
MAX_RETRIES: int = 3

# 대화 저장 파일이 위치할 디렉터리명 (프로젝트 루트 기준 상대 경로)
DATA_DIR: str = "data"

# 대화 저장 파일명에 사용할 타임스탬프 포맷 (예: 20260715_213000)
SAVE_FORMAT: str = "%Y%m%d_%H%M%S"

# Tavily 검색 설정

# Tavily 검색 API 기본 반환 결과 수
TAVILY_DEFAULT_MAX_RESULTS: int = 5

# Tavily 검색 깊이 — 일반 검색 시 사용 (basic: 빠르고 저렴, advanced: 깊은 검색)
TAVILY_DEFAULT_SEARCH_DEPTH: str = "basic"

# Tavily 검색 깊이 — 심화 검색 시 사용
TAVILY_ADVANCED_SEARCH_DEPTH: str = "advanced"

# 검색 쿼리 최적화: 사용자 발화에서 제거할 한국어 표현 (불필요한 조사·요청어)
QUERY_REMOVE_PHRASES: list[str] = [
    "알려줘",
    "알려주세요",
    "설명해줘",
    "설명해주세요",
    "찾아줘",
    "찾아주세요",
    "검색해줘",
    "검색해주세요",
    "에 대해",
    "에 관해",
    "에 대해서",
    "좀",
    "좀요",
    "해줘",
    "해주세요",
    "알고 싶어",
    "알고 싶어요",
    "궁금해",
    "궁금해요",
]

# 검색 쿼리 최적화: 연도로 변환할 시간 표현 (최신성 필터·쿼리 보강용)
TIME_INDICATOR_PHRASES: list[str] = [
    "최신",
    "최근",
    "요즘",
    "현재",
    "올해",
    "금년",
    "이번",
    "지금",
]
