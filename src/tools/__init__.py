"""도구(Tools) 모듈 패키지.

리서치 어시스턴트가 사용하는 외부 도구(웹 검색 등)와
OpenAI Function Calling용 도구 정의를 제공한다.

포함 모듈:
    web_search: Tavily API 기반 웹 검색 및 결과 포맷팅
    tool_definitions: OpenAI Function Calling JSON 스키마 정의
"""

from .tool_definitions import (
    AVAILABLE_TOOLS,
    SEARCH_WEB_TOOL,
    TOOLS_BY_NAME,
    get_all_tool_names,
    get_tool_by_name,
)
from .web_search import (
    SearchResult,
    format_search_result_for_llm,
    optimize_search_query,
    tavily_search,
    tavily_search_with_context,
)

__all__ = [
    # web_search
    "SearchResult",
    "format_search_result_for_llm",
    "optimize_search_query",
    "tavily_search",
    "tavily_search_with_context",
    # tool_definitions
    "AVAILABLE_TOOLS",
    "SEARCH_WEB_TOOL",
    "TOOLS_BY_NAME",
    "get_all_tool_names",
    "get_tool_by_name",
]
