"""Tavily API를 이용한 웹 검색 도구 모듈.

외부 웹 검색 결과를 수집·정리하고, SearchResult 데이터 클래스로
구조화된 형태로 반환하기 위해 사용한다.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from config import settings

logger = logging.getLogger(__name__)

load_dotenv()

_tavily_client = None


@dataclass
class SearchResult:
    """웹 검색 결과를 담는 데이터 클래스.

    Attributes:
        query: 실행한 검색 쿼리.
        answer: Tavily AI 요약 답변 (없을 수 있음).
        results: 검색 결과 딕셔너리 리스트 (title, url, content 등).
        sources: 출처 URL 리스트.
        search_time: 검색 소요 시간(초).
        raw_response: Tavily API 원본 응답.
    """

    query: str
    answer: Optional[str] = None
    results: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    search_time: float = 0.0
    raw_response: Optional[Dict[str, Any]] = None

    @property
    def result_count(self) -> int:
        """검색 결과 개수를 반환한다."""
        return len(self.results)

    @property
    def has_answer(self) -> bool:
        """AI 요약 답변(answer) 존재 여부를 반환한다."""
        return self.answer is not None and bool(self.answer.strip())

    def get_top_results(self, n: int = 3) -> List[Dict[str, Any]]:
        """상위 n개의 검색 결과를 반환한다."""
        return self.results[:n]

    def get_sources_as_string(self, separator: str = "\n") -> str:
        """출처 URL 리스트를 구분자로 연결한 문자열로 반환한다."""
        return separator.join(self.sources)


def _get_tavily_client(api_key: Optional[str] = None):
    """TavilyClient 싱글톤 인스턴스를 반환한다.

    최초 호출 시 lazy import로 TavilyClient를 로드하고,
    이후 호출에서는 캐시된 인스턴스를 재사용한다.
    api_key가 명시되면 캐시 없이 새 클라이언트를 반환한다.

    Args:
        api_key: Tavily API 키. None이면 환경변수 TAVILY_API_KEY를 사용한다.

    Returns:
        TavilyClient: API 키가 설정된 Tavily 클라이언트.

    Raises:
        ImportError: tavily-python 패키지가 설치되지 않은 경우.
    """
    global _tavily_client

    try:
        from tavily import TavilyClient
    except ImportError as exc:
        raise ImportError(
            "tavily-python 패키지가 설치되어 있지 않습니다. "
            "'pip install tavily-python' 명령으로 설치해주세요."
        ) from exc

    if api_key is not None:
        return TavilyClient(api_key=api_key)

    if _tavily_client is not None:
        return _tavily_client

    _tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    logger.debug("TavilyClient 초기화 완료")
    return _tavily_client


def tavily_search(
    query: str,
    api_key: Optional[str] = None,
    search_depth: str = "basic",
    include_answer: bool = True,
    include_raw_content: bool = False,
    max_results: int = 5,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
) -> SearchResult:
    """Tavily API로 웹 검색을 수행하고 구조화된 결과를 반환한다.

    Args:
        query: 검색할 질의 문자열.
        api_key: Tavily API 키. None이면 환경변수 TAVILY_API_KEY를 사용한다.
        search_depth: 검색 깊이 ("basic" 또는 "advanced").
        include_answer: Tavily AI 요약 답변 포함 여부.
        include_raw_content: 원본 HTML/텍스트 포함 여부.
        max_results: 반환할 최대 결과 수 (1~10).
        include_domains: 검색 대상으로 제한할 도메인 리스트.
        exclude_domains: 검색에서 제외할 도메인 리스트.

    Returns:
        SearchResult: 파싱된 검색 결과 객체.

    Raises:
        ValueError: query가 비어 있거나, API 키가 없거나, max_results가
            허용 범위를 벗어난 경우.
        ImportError: tavily-python 패키지가 설치되지 않은 경우.
        Exception: Tavily API 호출 중 발생한 기타 오류.

    Example:
        >>> result = tavily_search("Python asyncio tutorial")
        >>> print(result.result_count)
        5
        >>> print(result.has_answer)
        True
    """
    if not query or not query.strip():
        raise ValueError(
            "검색 쿼리가 비어 있습니다. 검색할 내용을 입력해주세요."
        )
    query = query.strip()

    resolved_api_key = api_key or os.getenv("TAVILY_API_KEY")
    if not resolved_api_key:
        raise ValueError(
            "Tavily API 키가 설정되지 않았습니다. "
            "api_key 파라미터를 전달하거나 .env 파일에 TAVILY_API_KEY를 "
            "추가해주세요."
        )

    if not 1 <= max_results <= 10:
        raise ValueError(
            f"max_results는 1~10 사이여야 합니다. (입력값: {max_results})"
        )

    logger.info(
        "Tavily 검색 시작: query=%r, depth=%s, max_results=%d",
        query,
        search_depth,
        max_results,
    )

    try:
        client = _get_tavily_client(resolved_api_key)

        search_params: Dict[str, Any] = {
            "query": query,
            "search_depth": search_depth,
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
            "max_results": max_results,
        }
        if include_domains:
            search_params["include_domains"] = include_domains
        if exclude_domains:
            search_params["exclude_domains"] = exclude_domains

        start_time = time.time()
        response = client.search(**search_params)
        elapsed = time.time() - start_time

        results: List[Dict[str, Any]] = response.get("results", [])
        sources = [
            item["url"] for item in results if item.get("url")
        ]

        search_result = SearchResult(
            query=query,
            answer=response.get("answer"),
            results=results,
            sources=sources,
            search_time=elapsed,
            raw_response=response,
        )

        logger.info(
            "Tavily 검색 완료: results=%d, elapsed=%.2fs",
            search_result.result_count,
            elapsed,
        )
        return search_result

    except Exception:
        logger.exception("Tavily 검색 실패: query=%r", query)
        raise


# settings에 없는 추가 제거 표현
_EXTRA_QUERY_REMOVE_PHRASES: List[str] = [
    "조사해줘",
    "조사해주세요",
    "뭐야",
    "무엇인가요",
    "관해서",
]

_CONTENT_MAX_LENGTH = 300


def tavily_search_with_context(
    query: str,
    context: Optional[str] = None,
    api_key: Optional[str] = None,
    max_results: int = 5,
) -> SearchResult:
    """컨텍스트를 포함해 Tavily 심층 검색을 수행한다.

    context가 주어지면 ``"{context} - {query}"`` 형태로 쿼리를 확장하고,
    search_depth는 ``advanced``로 고정하여 tavily_search를 호출한다.

    Args:
        query: 검색할 질의 문자열.
        context: 검색 맥락(주제·배경). 있으면 쿼리 앞에 붙인다.
        api_key: Tavily API 키. None이면 환경변수를 사용한다.
        max_results: 반환할 최대 결과 수 (1~10).

    Returns:
        SearchResult: 파싱된 검색 결과 객체.

    Raises:
        ValueError: query가 비어 있거나 API 키·max_results가 유효하지 않은 경우.
        ImportError: tavily-python 패키지가 설치되지 않은 경우.
        Exception: Tavily API 호출 중 발생한 기타 오류.

    Example:
        >>> result = tavily_search_with_context(
        ...     "LangChain RAG",
        ...     context="Python AI 프레임워크",
        ... )
        >>> "Python AI 프레임워크" in result.query
        True
    """
    search_query = f"{context.strip()} - {query}" if context else query
    return tavily_search(
        query=search_query,
        api_key=api_key,
        search_depth=settings.TAVILY_ADVANCED_SEARCH_DEPTH,
        max_results=max_results,
    )


def format_search_result_for_llm(search_result: SearchResult) -> str:
    """검색 결과를 LLM 컨텍스트용 마크다운 문자열로 변환한다.

    Args:
        search_result: 포맷할 SearchResult 객체.

    Returns:
        str: LLM에 전달할 마크다운 형식의 검색 결과 문자열.

    Example:
        >>> result = SearchResult(
        ...     query="Python",
        ...     answer="Python is a programming language.",
        ...     results=[{"title": "Python.org", "url": "https://python.org",
        ...               "score": 0.9, "content": "Official site."}],
        ...     sources=["https://python.org"],
        ...     search_time=1.23,
        ... )
        >>> formatted = format_search_result_for_llm(result)
        >>> "웹 검색 결과" in formatted
        True
    """
    lines: List[str] = [
        f"## 웹 검색 결과: '{search_result.query}'",
        f"검색 시간: {search_result.search_time:.2f}초",
        "",
    ]

    if search_result.has_answer:
        lines.extend([
            "### 📋 요약",
            search_result.answer or "",
            "",
        ])

    lines.append("### 📰 상세 검색 결과")
    for index, item in enumerate(search_result.results, start=1):
        title = item.get("title", "제목 없음")
        url = item.get("url", "")
        score = item.get("score", "N/A")
        content = item.get("content", "")
        if len(content) > _CONTENT_MAX_LENGTH:
            content = content[:_CONTENT_MAX_LENGTH] + "..."

        lines.extend([
            f"**[{index}] {title}**",
            f"- 출처: {url}",
            f"- 관련도: {score}",
            f"- 내용: {content}",
            "",
        ])

    if search_result.sources:
        lines.append("### 📚 참고 출처")
        for index, source in enumerate(search_result.sources, start=1):
            lines.append(f"{index}. {source}")

    return "\n".join(lines)


def optimize_search_query(user_input: str) -> str:
    """사용자 입력을 검색에 최적화된 쿼리로 변환한다.

    불필요한 한국어 요청 표현을 제거하고, 시간 표현이 포함된 경우
    현재 연도를 추가하여 검색 정확도를 높인다.

    Args:
        user_input: 사용자의 원본 입력 문자열.

    Returns:
        str: 최적화된 검색 쿼리.

    Example:
        >>> optimize_search_query("최신 Python asyncio 알려줘")
        'Python asyncio 2026'
    """
    query = user_input.strip()

    remove_phrases = settings.QUERY_REMOVE_PHRASES + _EXTRA_QUERY_REMOVE_PHRASES
    for phrase in sorted(remove_phrases, key=len, reverse=True):
        query = query.replace(phrase, "")

    has_time_indicator = False
    for phrase in settings.TIME_INDICATOR_PHRASES:
        if phrase in query:
            has_time_indicator = True
            query = query.replace(phrase, "")

    if has_time_indicator:
        query = f"{query} {datetime.now().year}"

    return " ".join(query.split())
