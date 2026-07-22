"""OpenAI Function Calling에 사용할 도구 정의 JSON 스키마.

각 도구의 ``description``은 LLM이 도구 사용 여부를 판단하는 핵심
근거가 되므로, 사용해야/하지 않아야 하는 경우를 구체적으로 기술해야
한다. 모호한 설명은 불필요한 도구 호출이나 누락으로 이어질 수 있다.
"""

from typing import Any

SEARCH_WEB_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": """웹에서 최신 정보를 검색합니다.

이 도구를 사용해야 하는 경우:
- 실시간 정보 (주가, 날씨, 환율, 스포츠 결과 등)
- 최근 뉴스나 이벤트
- 특정 사실의 확인/검증
- "검색해줘", "찾아줘", "조사해줘" 등 명시적 요청
- 2024년 이후 정보
- 기업/인물/제품의 최신 현황
- 트렌드, 동향, 전망

이 도구를 사용하지 않아야 하는 경우:
- 일반 개념/정의 설명
- 프로그래밍 문법, 코드 작성
- 수학 계산, 논리적 추론
- 개인적 조언, 의견
- 창작물 (시, 소설, 이메일 등)
- 번역
- 확정된 역사적 사실""",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "검색할 키워드나 질문. 구체적으로 작성. "
                        "예: 'Tesla stock 2024', 'AI agent trends'"
                    ),
                },
                "search_depth": {
                    "type": "string",
                    "enum": ["basic", "advanced"],
                    "description": (
                        "basic: 빠른 일반 검색, advanced: 심층 검색. "
                        "기본값 basic"
                    ),
                },
            },
            "required": ["query"],
        },
    },
}

# 향후 추가 예정:
# FETCH_WEBPAGE_TOOL — 특정 URL의 웹페이지 본문 추출
# CALCULATOR_TOOL — 수식 계산
# CODE_EXECUTOR_TOOL — 코드 실행 및 결과 반환

AVAILABLE_TOOLS: list[dict[str, Any]] = [
    SEARCH_WEB_TOOL,
]

TOOLS_BY_NAME: dict[str, dict[str, Any]] = {
    "search_web": SEARCH_WEB_TOOL,
}


def get_tool_by_name(name: str) -> dict[str, Any]:
    """도구 이름으로 Function Calling 스키마를 반환한다.

    Args:
        name: 도구 이름 (예: ``"search_web"``).

    Returns:
        dict: 해당 도구의 JSON 스키마.

    Raises:
        KeyError: 등록되지 않은 도구 이름인 경우.
    """
    return TOOLS_BY_NAME[name]


def get_all_tool_names() -> list[str]:
    """등록된 모든 도구 이름을 반환한다.

    Returns:
        list[str]: 사용 가능한 도구 이름 리스트.
    """
    return list(TOOLS_BY_NAME.keys())
