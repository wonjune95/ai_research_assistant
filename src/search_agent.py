"""웹 검색 도구들을 오케스트레이션하는 상위 레벨 에이전트 모듈.

src/tools/web_search.py의 개별 함수(쿼리 최적화, Tavily 검색, LLM용 포맷팅)를
조합해 "검색 한 번"이라는 단위 작업으로 묶어 제공한다. 검색 이력을 보관해
후속 질문에서 맥락을 이어갈 수 있도록 하는 것도 이 클래스의 역할이다.
"""

import logging
from typing import List, Optional

from src.tools.web_search import (
    SearchResult,
    format_search_result_for_llm,
    optimize_search_query,
    tavily_search,
    tavily_search_with_context,
)

logger = logging.getLogger(__name__)

# max_results 허용 범위 (Tavily API 제한과 동일하게 맞춘다)
MIN_MAX_RESULTS = 1
MAX_MAX_RESULTS = 10


class SearchAgent:
    """웹 검색 도구들을 조합해 검색 작업을 수행하는 에이전트.

    Attributes:
        max_results: 검색 시 기본으로 요청할 결과 수 (1~10).
        optimize_queries: 사용자 입력을 검색용 쿼리로 최적화할지 여부.
        search_history: 지금까지 수행한 검색의 SearchResult 목록.

    Example:
        >>> agent = SearchAgent(max_results=3)
        >>> agent.max_results
        3
        >>> agent.optimize_queries
        True
        >>> agent.search_history
        []
    """

    def __init__(
        self,
        max_results: int = 5,
        optimize_queries: bool = True,
    ) -> None:
        """SearchAgent를 초기화합니다.

        Args:
            max_results: 검색 시 기본으로 요청할 결과 수. 1~10 사이여야 한다.
            optimize_queries: True이면 검색 전에 optimize_search_query()로
                사용자 입력을 정제한다.

        Raises:
            ValueError: max_results가 1~10 범위를 벗어난 경우.
        """
        if not MIN_MAX_RESULTS <= max_results <= MAX_MAX_RESULTS:
            raise ValueError(
                f"max_results는 {MIN_MAX_RESULTS}~{MAX_MAX_RESULTS} 사이여야 "
                f"합니다. (입력값: {max_results})"
            )

        self.max_results: int = max_results
        self.optimize_queries: bool = optimize_queries
        self.search_history: List[SearchResult] = []

        logger.info(
            "SearchAgent 초기화 완료 (max_results=%d, optimize_queries=%s)",
            max_results,
            optimize_queries,
        )

    def search(
        self,
        query: str,
        search_depth: str = "basic",
        include_answer: bool = True,
        max_results: Optional[int] = None,
        optimize_query: Optional[bool] = None,
    ) -> SearchResult:
        """웹 검색을 수행하고 결과를 히스토리에 기록합니다.

        optimize_query가 활성화된 경우 검색 직전에 쿼리를 정제하지만,
        반환되는 SearchResult.query에는 사용자가 입력한 원본 쿼리를 담는다.
        호출자(예: 대화 히스토리·출처 표시)가 "무엇을 물었는지"를 그대로
        볼 수 있어야 하기 때문이다.

        Args:
            query: 검색할 질의 문자열.
            search_depth: 검색 깊이 ("basic" 또는 "advanced").
            include_answer: Tavily AI 요약 답변 포함 여부.
            max_results: 반환할 최대 결과 수. None이면 self.max_results 사용.
            optimize_query: 쿼리 최적화 여부. None이면 self.optimize_queries 사용.

        Returns:
            SearchResult: 검색 결과 객체 (query는 원본 쿼리).

        Raises:
            ValueError: query가 비어 있거나 max_results가 유효 범위를 벗어난 경우.
            ImportError: tavily-python 패키지가 설치되지 않은 경우.
            Exception: Tavily API 호출 중 발생한 기타 오류.

        Example:
            >>> agent = SearchAgent()
            >>> result = agent.search("Python asyncio 최신 동향 알려줘")
            >>> result.query
            'Python asyncio 최신 동향 알려줘'
        """
        if not query or not query.strip():
            raise ValueError(
                "검색 쿼리가 비어 있습니다. 검색할 내용을 입력해주세요."
            )

        original_query = query.strip()
        resolved_max_results = (
            max_results if max_results is not None else self.max_results
        )
        should_optimize = (
            optimize_query if optimize_query is not None else self.optimize_queries
        )

        search_query = original_query
        if should_optimize:
            search_query = optimize_search_query(original_query)
            if search_query != original_query:
                logger.info(
                    "쿼리 최적화: %r -> %r", original_query, search_query
                )

        result = tavily_search(
            query=search_query,
            search_depth=search_depth,
            include_answer=include_answer,
            max_results=resolved_max_results,
        )

        # 최적화된 쿼리 대신 사용자가 실제로 입력한 원본 쿼리를 남긴다.
        result.query = original_query
        self.search_history.append(result)

        logger.info(
            "검색 완료 (query=%r, results=%d, 누적 검색 %d회)",
            original_query,
            result.result_count,
            len(self.search_history),
        )
        return result

    def search_with_context(
        self,
        query: str,
        context: str,
        max_results: Optional[int] = None,
    ) -> SearchResult:
        """맥락을 덧붙여 심층 검색을 수행하고 결과를 히스토리에 기록합니다.

        search()와 달리 쿼리 최적화를 적용하지 않는다. 맥락이 이미 주어진
        상황에서는 원문 표현을 그대로 유지하는 편이 검색 정확도에 유리하다.

        Args:
            query: 검색할 질의 문자열.
            context: 검색 맥락(주제·배경). 쿼리 앞에 붙는다.
            max_results: 반환할 최대 결과 수. None이면 self.max_results 사용.

        Returns:
            SearchResult: 검색 결과 객체 (query는 맥락이 결합된 쿼리).

        Raises:
            ValueError: query가 비어 있거나 max_results가 유효 범위를 벗어난 경우.
            ImportError: tavily-python 패키지가 설치되지 않은 경우.
            Exception: Tavily API 호출 중 발생한 기타 오류.

        Example:
            >>> agent = SearchAgent()
            >>> result = agent.search_with_context(
            ...     "RAG 구현 방법", context="LangChain"
            ... )
            >>> "LangChain" in result.query
            True
        """
        resolved_max_results = (
            max_results if max_results is not None else self.max_results
        )

        result = tavily_search_with_context(
            query=query,
            context=context,
            max_results=resolved_max_results,
        )

        self.search_history.append(result)

        logger.info(
            "맥락 검색 완료 (query=%r, context=%r, results=%d, 누적 검색 %d회)",
            query,
            context,
            result.result_count,
            len(self.search_history),
        )
        return result

    def format_for_llm(self, search_result: SearchResult) -> str:
        """검색 결과를 LLM 컨텍스트용 마크다운 문자열로 변환합니다.

        Args:
            search_result: 포맷할 SearchResult 객체.

        Returns:
            str: LLM에 전달할 마크다운 형식의 검색 결과 문자열.
        """
        return format_search_result_for_llm(search_result)

    def get_sources(self) -> List[str]:
        """가장 최근 검색의 출처 URL 목록을 반환합니다.

        Returns:
            List[str]: 마지막 검색의 출처 URL 리스트.
                검색 이력이 없으면 빈 리스트.
        """
        if not self.search_history:
            return []
        return self.search_history[-1].sources

    def get_last_result(self) -> Optional[SearchResult]:
        """가장 최근 검색 결과를 반환합니다.

        Returns:
            Optional[SearchResult]: 마지막 SearchResult. 검색 이력이 없으면 None.
        """
        if not self.search_history:
            return None
        return self.search_history[-1]

    def get_search_count(self) -> int:
        """지금까지 수행한 검색 횟수를 반환합니다.

        Returns:
            int: search_history에 쌓인 검색 결과 개수.
        """
        return len(self.search_history)

    def clear_history(self) -> None:
        """검색 히스토리를 모두 비웁니다."""
        cleared_count = len(self.search_history)
        self.search_history.clear()
        logger.info("검색 히스토리 초기화 완료 (%d건 삭제)", cleared_count)
