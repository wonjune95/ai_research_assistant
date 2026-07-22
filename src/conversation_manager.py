"""OpenAI Chat Completions API를 이용해 대화 히스토리를 관리하는 모듈."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AuthenticationError, OpenAI, OpenAIError
from openai.types.chat import ChatCompletion

from config import settings
from config.prompts import (
    RESEARCH_ASSISTANT_SYSTEM_MESSAGE,
    RESEARCH_ASSISTANT_SYSTEM_MESSAGE_V2,
)
from src.exceptions import ConfigurationError, ConversationLoadError

# ---------------------------------------------------------------------------
# 내부 모듈 — 웹 검색(Function Calling) 도구
#
# SearchAgent나 도구 정의가 아직 없는 환경에서도 이 모듈 자체는 import될 수
# 있도록 방어한다. 실패하면 TOOLS_AVAILABLE이 False가 되고, 도구 기능만
# 비활성화된 채 일반 대화는 그대로 동작한다.
# ---------------------------------------------------------------------------
try:
    from src.search_agent import SearchAgent
    from src.tools.tool_definitions import AVAILABLE_TOOLS

    TOOLS_AVAILABLE = True
except ImportError as _import_error:
    # logger는 아래에서 정의되므로 여기서는 모듈 로거를 직접 가져와 기록한다.
    logging.getLogger(__name__).warning(
        "웹 검색 도구를 불러오지 못해 Function Calling이 비활성화됩니다: %s",
        _import_error,
    )
    SearchAgent = None  # type: ignore[assignment, misc]
    AVAILABLE_TOOLS: list[dict[str, Any]] = []  # type: ignore[no-redef]
    TOOLS_AVAILABLE = False

load_dotenv()

logger = logging.getLogger(__name__)

# 시스템 프롬프트 기본값. 도구가 활성화된 구성에서는 검색 지침이 포함된 V2를
# 쓰고, 도구 없이 순수 대화만 할 때는 V1을 쓴다.
DEFAULT_SYSTEM_MESSAGE: str = RESEARCH_ASSISTANT_SYSTEM_MESSAGE_V2
DEFAULT_SYSTEM_MESSAGE_NO_TOOLS: str = RESEARCH_ASSISTANT_SYSTEM_MESSAGE

# ConversationManager 기본 설정값 (모델명/temperature/재시도 횟수는
# config.settings에서 가져온다. 요약 전용 temperature는 이 모듈에서만
# 쓰이는 값이라 로컬 상수로 둔다.)
SUMMARY_TEMPERATURE = 0.3

# determine_state / chat이 사용하는 대화 상태 상수
STATE_IDLE = "idle"
STATE_RESPONDING = "responding"
STATE_RESEARCHING = "researching"
# 예상치 못한 예외로 응답 생성이 중단된 상태. 예외를 그대로 올리기 전에
# 남겨두어, 호출부가 관리자 상태를 확인해 원인을 파악할 수 있게 한다.
STATE_ERROR = "error"

# summarize_conversation이 사용하는 상수
MIN_MESSAGES_FOR_SUMMARY = 3
SHORT_CONVERSATION_MESSAGE = "대화가 충분히 길지 않습니다"
SUMMARY_REQUEST_PROMPT = "지금까지의 대화를 3문장으로 요약해주세요."

# chat()과 summarize_conversation() 양쪽에서 공통으로 쓰는 인증 실패 안내 메시지.
# 재시도해도 해결되지 않는 오류이므로 문구를 한 곳에서만 관리한다.
AUTH_ERROR_MESSAGE = (
    "OpenAI API 키가 유효하지 않습니다. .env 파일의 OPENAI_API_KEY 값을 확인해주세요."
)

# Function Calling 관련 상수
# 도구 호출 -> 결과 전달 -> 재호출 왕복을 허용할 최대 횟수. 모델이 도구 호출만
# 반복하며 최종 답변을 내지 않는 상황을 막기 위한 안전장치다.
MAX_TOOL_ITERATIONS = 3

# 현재 디스패처가 지원하는 도구 이름 (config/tool_definitions.py의 정의와 일치)
TOOL_SEARCH_WEB = "search_web"


class ConversationManager:
    """OpenAI Chat Completions API를 이용해 대화 히스토리를 관리하는 클래스.

    Attributes:
        client: OpenAI API 호출에 사용하는 클라이언트.
        model: 사용할 OpenAI 모델 이름.
        messages: 지금까지 주고받은 대화 히스토리
            (system/user/assistant/tool 메시지).
        turn_count: 성공적으로 완료된 대화(사용자 질문-AI 응답 쌍)의 횟수.
        state: 현재 대화 상태 (STATE_IDLE | STATE_RESPONDING | STATE_RESEARCHING).
        enable_search: Function Calling(웹 검색 도구) 사용 여부.
        search_agent: 웹 검색을 수행할 SearchAgent. enable_search가 False면 None.
        tools: 모델에 노출할 도구 스키마 목록. enable_search가 False면 None.
    """

    # determine_state에서 사용하는 리서치 모드 감지 키워드.
    # NOTE: 4주차에 LLM 기반 판단(의도 분류)으로 고도화 예정. 현재는 단순
    # 키워드 포함 여부로만 판단하는 1주차 수준의 규칙 기반 구현이다.
    RESEARCH_KEYWORDS: list[str] = ["조사", "분석", "리서치", "알아봐", "찾아봐"]

    def __init__(
        self,
        system_message: str | None = None,
        model: str = settings.DEFAULT_MODEL,
        enable_search: bool = True,
    ) -> None:
        """ConversationManager를 초기화합니다.

        Args:
            system_message: 대화의 시스템 프롬프트. 주어지면 messages의
                첫 항목으로 추가됩니다.
            model: 사용할 OpenAI 모델 이름. 기본값은 config.settings.DEFAULT_MODEL.
            enable_search: True이면 웹 검색 도구를 모델에 노출해 Function Calling을
                활성화한다. 검색 없이 순수 대화만 하려면 False로 둔다.

        Raises:
            ConfigurationError: OPENAI_API_KEY 환경 변수가 설정되어 있지
                않거나, OpenAI 클라이언트 생성에 실패한 경우.
        """
        # API 키가 아예 없는 경우를 가장 흔한 설정 실수로 보고 먼저 명확히 안내한다.
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY 환경 변수가 설정되어 있지 않습니다.")
            raise ConfigurationError(
                "OpenAI API 키가 설정되지 않았습니다. .env 파일에 "
                "OPENAI_API_KEY=sk-... 형태로 값을 추가해주세요."
            )

        try:
            self.client: OpenAI = OpenAI(api_key=api_key)
        except OpenAIError as error:
            # 키는 있지만 형식이 잘못되었거나 클라이언트 설정 자체가 실패한 경우
            logger.error("OpenAI 클라이언트 생성 실패: %s", error)
            raise ConfigurationError(
                "OpenAI 클라이언트를 초기화하지 못했습니다. API 키 형식을 확인해주세요."
            ) from error

        self.model: str = model
        # tool 흐름에서는 값이 문자열이 아닌 항목(assistant의 tool_calls 리스트 등)이
        # 들어오므로 값 타입을 Any로 둔다.
        self.messages: list[dict[str, Any]] = []
        self.turn_count: int = 0
        self.state: str = STATE_IDLE

        # 도구 모듈 import에 실패한 환경에서는 요청과 무관하게 검색을 끈다.
        if enable_search and not TOOLS_AVAILABLE:
            logger.warning(
                "enable_search=True로 요청되었지만 도구 모듈을 사용할 수 없어 "
                "웹 검색 없이 동작합니다."
            )
        self.enable_search: bool = enable_search and TOOLS_AVAILABLE

        # SearchAgent 생성 자체는 Tavily API 키를 요구하지 않는다. 키가 없으면
        # 실제 검색 시점에 실패하고, 그 오류는 _execute_tool에서 도구 결과로
        # 변환되어 모델이 사용자에게 상황을 설명할 수 있게 된다.
        self.search_agent: "SearchAgent | None" = (
            SearchAgent() if self.enable_search else None
        )
        # 매 호출마다 AVAILABLE_TOOLS를 참조하지 않도록 인스턴스에 보관한다.
        self.tools: list[dict[str, Any]] | None = (
            AVAILABLE_TOOLS if self.enable_search else None
        )

        if system_message:
            self.messages.append({"role": "system", "content": system_message})

        logger.info(
            "ConversationManager 초기화 완료 (model=%s, enable_search=%s)",
            model,
            self.enable_search,
        )

    def determine_state(self, user_input: str) -> str:
        """사용자 입력을 바탕으로 대화 상태를 판단합니다.

        키워드 기반의 단순 규칙으로 RESEARCH_KEYWORDS 중 하나라도 포함되면
        리서치 모드로 판단한다.

        NOTE: 4주차에 LLM 기반 판단으로 고도화 예정.

        Args:
            user_input: 사용자가 입력한 메시지.

        Returns:
            STATE_RESEARCHING 또는 STATE_RESPONDING.
        """
        state = STATE_RESEARCHING if any(
            keyword in user_input for keyword in self.RESEARCH_KEYWORDS
        ) else STATE_RESPONDING
        logger.debug("determine_state(%r) -> %s", user_input, state)
        return state

    def _call_api_with_tools(self, include_tools: bool = True) -> Any:
        """도구 정의를 포함하여 OpenAI API를 호출합니다.

        검색 기능이 활성화되어 있으면 tools 파라미터를 추가하여
        LLM이 필요시 도구를 호출할 수 있도록 합니다.

        tool_choice는 "auto"로 지정한다. 다른 선택지는 다음과 같다.
            - "auto": LLM이 도구 사용 여부를 자동 판단 (기본값)
            - "required": 반드시 도구를 사용
            - "none": 도구를 사용하지 않음

        이 메서드는 재시도를 하지 않는 단일 호출이다. 재시도가 필요하면
        이 메서드를 감싸는 _call_with_retry()를 사용한다.

        Args:
            include_tools: False이면 검색이 활성화되어 있어도 도구를 노출하지
                않는다. chat()의 마지막 왕복에서 모델이 도구 호출만 반복하지
                않고 반드시 텍스트로 답하도록 강제할 때 사용한다.

        Returns:
            OpenAI API 응답 객체.
        """
        call_params: dict[str, Any] = {
            "model": self.model,
            "messages": self.messages,
            "temperature": settings.DEFAULT_TEMPERATURE,
            "max_tokens": settings.MAX_TOKENS,
        }

        if include_tools and self.enable_search and self.tools:
            call_params["tools"] = self.tools
            call_params["tool_choice"] = "auto"  # LLM이 자동 판단

        response = self.client.chat.completions.create(**call_params)
        return response

    def _call_with_retry(
        self,
        max_retries: int,
        include_tools: bool = True,
    ) -> tuple[ChatCompletion | None, str | None]:
        """_call_api_with_tools()를 호출하고, 실패 시 지수 백오프로 재시도합니다.

        attempt은 0부터 시작하며 "이번이 몇 번째 시도인지"를 의미한다.
        - attempt 0: 최초 시도 (재시도 아님, 대기 없음)
        - attempt 1~max_retries: 재시도. 대기 시간은 2 ** attempt 초
          (1차 재시도 attempt=1 -> 2초, 2차 attempt=2 -> 4초, 3차 attempt=3 -> 8초)

        인증 실패(AuthenticationError)는 재시도해도 결과가 바뀌지 않으므로
        백오프 없이 즉시 중단한다.

        Args:
            max_retries: 재시도할 최대 횟수.
            include_tools: 도구 노출 여부. _call_api_with_tools()에 그대로 전달된다.

        Returns:
            성공하면 (ChatCompletion, None), 실패하면
            (None, 사용자 친화적인 에러 메시지) 튜플.
        """
        last_error: OpenAIError | None = None
        for attempt in range(max_retries + 1):
            try:
                logger.debug("OpenAI API 호출 시도 %d/%d", attempt + 1, max_retries + 1)
                response = self._call_api_with_tools(include_tools=include_tools)
                return response, None
            except AuthenticationError as error:
                logger.error("OpenAI 인증 실패 - API 키를 확인해주세요: %s", error)
                return None, AUTH_ERROR_MESSAGE
            except OpenAIError as error:
                # 레이트 리밋, 일시적 연결 오류, 서버 오류 등 재시도로 회복 가능한 오류
                last_error = error
                logger.warning(
                    "OpenAI API 호출 실패 (시도 %d/%d): %s",
                    attempt + 1,
                    max_retries + 1,
                    error,
                )
                # 아직 재시도 기회가 남아있으면 대기 후 다시 시도
                if attempt < max_retries:
                    wait_seconds = 2 ** (attempt + 1)
                    print(
                        f"일시적인 오류가 발생했습니다. {wait_seconds}초 후 "
                        f"재시도합니다... ({attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_seconds)

        logger.error(
            "OpenAI API 호출 최종 실패 (총 %d회 시도): %s", max_retries + 1, last_error
        )
        return None, (
            f"죄송합니다, {max_retries}번 재시도했지만 응답을 가져오지 못했습니다. "
            "잠시 후 다시 시도해주세요."
        )

    @staticmethod
    def _tool_error(message: str) -> str:
        """도구 실행 실패를 JSON 문자열로 변환합니다.

        오류도 정상 결과와 마찬가지로 tool 메시지의 content로 모델에 전달된다.
        예외를 그대로 올리면 대화 자체가 끊기므로, 모델이 실패를 인지하고
        사용자에게 설명할 수 있도록 문자열로 바꿔 돌려준다.

        Args:
            message: 사용자/모델에게 전달할 오류 설명.

        Returns:
            ``{"error": ...}`` 형태의 JSON 문자열.
        """
        return json.dumps({"error": message}, ensure_ascii=False)

    def _execute_tool(self, function_name: str, arguments: dict[str, Any]) -> str:
        """지정된 도구를 실행하고 결과를 반환합니다.

        Args:
            function_name: 실행할 도구 이름.
            arguments: 도구에 전달할 인자 딕셔너리.
                (JSON 문자열 파싱은 호출부인 _handle_tool_calls가 담당한다)

        Returns:
            str: 도구 실행 결과 (LLM이 이해할 수 있는 형식).
                성공하면 format_for_llm()의 마크다운 문자열,
                실패하면 ``{"error": ...}`` 형태의 JSON 문자열.
        """
        try:
            if function_name == TOOL_SEARCH_WEB:
                query = arguments.get("query", "")
                search_depth = arguments.get(
                    "search_depth", settings.TAVILY_DEFAULT_SEARCH_DEPTH
                )

                # 쿼리 검증
                if not query:
                    logger.warning("검색어가 비어 있어 도구를 실행하지 않습니다.")
                    return self._tool_error("검색어가 비어있습니다.")

                if self.search_agent is None:
                    logger.error("검색이 비활성화된 상태에서 도구 호출이 요청됨")
                    return self._tool_error("웹 검색 도구가 비활성화되어 있습니다.")

                # SearchAgent로 검색 실행
                search_result = self.search_agent.search(
                    query=query,
                    search_depth=search_depth,
                )

                # LLM용 포맷으로 변환
                formatted = self.search_agent.format_for_llm(search_result)

                logger.info(
                    "검색 완료: %d개 결과, %.2f초",
                    search_result.result_count,
                    search_result.search_time,
                )
                return formatted

            logger.warning("알 수 없는 도구: %s", function_name)
            return self._tool_error(f"알 수 없는 도구: {function_name}")

        except Exception as error:
            # Tavily API 키 누락(ValueError), 패키지 미설치(ImportError),
            # 네트워크 오류 등 무엇이든 대화를 끊지 않고 모델에 넘긴다.
            logger.error("도구 실행 실패 (%s): %s", function_name, error)
            return self._tool_error(f"도구 실행 실패: {error}")

    def _handle_tool_calls(
        self,
        message: Any,
        max_retries: int = settings.MAX_RETRIES,
        depth: int = 0,
    ) -> str:
        """LLM이 요청한 도구 호출을 처리합니다.

        1. Assistant의 도구 호출 요청을 메시지에 저장
        2. 각 도구를 실행하고 결과를 메시지에 추가
        3. 도구 결과를 포함하여 API를 다시 호출
        4. 최종 응답을 반환

        재호출한 응답이 또 도구를 요청하면 자기 자신을 재귀 호출해 이어서
        처리한다. depth가 MAX_TOOL_ITERATIONS에 도달하면 도구를 노출하지 않고
        호출해 반드시 텍스트 답변이 나오도록 한다(무한 루프 방지).

        Args:
            message: OpenAI API 응답의 message 객체 (tool_calls 포함).
            max_retries: 재호출이 실패했을 때 재시도할 최대 횟수.
            depth: 현재 도구 호출 왕복 횟수 (재귀 시 내부적으로 증가).

        Returns:
            str: 도구 결과를 반영한 최종 AI 응답.
        """
        # 1. Assistant 메시지 저장 (tool_calls 포함)
        #    SDK 객체를 그대로 넣지 않고 필요한 필드만 추려 담는다.
        #    save_conversation()이 히스토리를 json.dumps()로 저장하므로
        #    모든 항목이 JSON 직렬화 가능해야 하기 때문이다.
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": message.content,  # None일 수 있음
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in message.tool_calls
            ],
        }
        self.messages.append(assistant_msg)

        # 2. 각 도구 호출 처리
        #    assistant의 tool_calls와 tool 응답은 반드시 1:1로 짝을 이뤄야 하므로
        #    실패하더라도 모든 tool_call에 대해 메시지를 남긴다.
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            raw_arguments = tool_call.function.arguments

            try:
                arguments: dict[str, Any] = (
                    json.loads(raw_arguments) if raw_arguments else {}
                )
            except json.JSONDecodeError as error:
                # 모델이 깨진 JSON을 생성한 경우. 예외를 올리면 대화가 끊기므로
                # 오류 문구를 도구 결과로 대신 전달한다.
                logger.error(
                    "도구 인자 JSON 파싱 실패 (%s): %s", raw_arguments, error
                )
                result = f"[도구 오류] 도구 인자를 해석하지 못했습니다: {error}"
            else:
                logger.info("도구 실행: %s(%s)", function_name, arguments)
                result = self._execute_tool(function_name, arguments)

            # 결과를 메시지에 추가 (tool role, 원본 tool_call_id 그대로 사용)
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

        # 3. 도구 결과를 포함하여 API 재호출
        #    마지막 왕복에서는 도구를 노출하지 않아 텍스트 답변을 강제한다.
        is_final_round = depth + 1 >= MAX_TOOL_ITERATIONS
        final_response, error_message = self._call_with_retry(
            max_retries, include_tools=not is_final_round
        )

        if error_message is not None:
            # 재호출에 실패했다. 이미 저장된 assistant/tool 메시지는 짝이 맞는
            # 완결된 상태이므로 히스토리를 되돌리지 않아도 다음 호출에 문제가 없다.
            return error_message

        assert final_response is not None
        final_message = final_response.choices[0].message

        # 재호출 응답이 또 도구를 요청하면 이어서 처리한다.
        if final_message.tool_calls:
            logger.info("추가 도구 호출 감지 (왕복 %d/%d)", depth + 2, MAX_TOOL_ITERATIONS)
            return self._handle_tool_calls(
                final_message, max_retries=max_retries, depth=depth + 1
            )

        # 4. 최종 응답 저장 및 반환
        final_content: str = final_message.content or ""
        self.messages.append({"role": "assistant", "content": final_content})
        return final_content

    def chat(self, user_input: str, max_retries: int = settings.MAX_RETRIES) -> str:
        """사용자 입력을 히스토리에 추가하고 모델 응답을 반환합니다.

        enable_search가 True이면 웹 검색 도구를 모델에 노출한다. 모델이 도구를
        호출하면 _handle_tool_calls()가 실행 결과를 히스토리에 덧붙이고 API를
        다시 호출해 최종 답변을 만들어 온다.

        API 호출이 일시적으로 실패하면 지수 백오프(exponential backoff)를
        적용해 최대 max_retries번까지 재시도한다 (대기 시간: 2초 -> 4초 -> 8초).

        Args:
            user_input: 사용자가 입력한 메시지.
            max_retries: API 호출 실패 시 재시도할 최대 횟수.
                기본값 config.settings.MAX_RETRIES.

        Returns:
            모델이 생성한 응답 텍스트. 재시도를 모두 소진하고도 실패하면
            사용자 친화적인 에러 메시지를 반환합니다.

        Raises:
            ValueError: user_input이 비어 있거나 공백뿐인 경우.
            Exception: 재시도로 회복할 수 없는 예상치 못한 오류가 발생한 경우.
                이때 state는 STATE_ERROR로 남는다.
        """
        # 1. 입력 검증
        if not user_input or not user_input.strip():
            raise ValueError("입력이 비어있습니다.")
        user_input = user_input.strip()

        # 2. 사용자 입력을 바탕으로 상태를 판단하고 갱신
        #    (idle -> responding/researching)
        self.state = self.determine_state(user_input)

        # 3. 이번 턴에서 추가되는 메시지의 시작 위치를 기록해 둔다.
        #    첫 호출이 실패하면 여기까지 잘라내 히스토리를 원래대로 되돌린다.
        history_checkpoint = len(self.messages)

        # 4. 사용자 입력을 히스토리에 추가
        self.messages.append({"role": "user", "content": user_input})
        logger.debug(
            "사용자 입력 히스토리에 추가 (state=%s, 현재 메시지 수=%d)",
            self.state,
            len(self.messages),
        )

        try:
            # 5. 도구를 포함해 API 호출 (실패 시 내부적으로 재시도)
            response, error_message = self._call_with_retry(max_retries)

            if error_message is not None:
                # 재시도로도 회복하지 못한 API 오류. 대화를 끊지 않고 안내 문구를
                # 돌려주는 기존 동작을 유지하되, 방금 추가한 사용자 메시지는
                # 되돌려 히스토리를 일관되게 유지한다.
                del self.messages[history_checkpoint:]
                self.state = STATE_IDLE
                return error_message

            assert response is not None
            message = response.choices[0].message

            # 6. 도구 호출 확인 및 처리
            #    tool_calls가 None이거나 빈 리스트면 일반 응답으로 처리한다.
            if message.tool_calls:
                logger.info("도구 호출 감지: %d개", len(message.tool_calls))
                self.state = STATE_RESEARCHING
                result = self._handle_tool_calls(message, max_retries=max_retries)
            else:
                # 일반 응답 (content가 없을 가능성에 대비해 빈 문자열로 방어)
                result = message.content or ""
                self.messages.append({"role": "assistant", "content": result})

            self.turn_count += 1
            self.state = STATE_IDLE
            logger.info("대화 응답 수신 완료 (turn_count=%d)", self.turn_count)
            return result

        except Exception as error:
            # API 오류는 위에서 이미 안내 문구로 처리되므로, 여기에 도달하는 것은
            # 예상하지 못한 오류다. 원인을 감추지 않도록 그대로 다시 올린다.
            self.state = STATE_ERROR
            logger.error("응답 생성 실패: %s", error)
            raise

    def summarize_conversation(self) -> str:
        """지금까지의 대화 히스토리를 3문장으로 요약합니다.

        요약 요청 메시지는 실제 대화 흐름에 영향을 주지 않도록 self.messages에는
        추가하지 않고, API 호출 시에만 임시로 덧붙여 전달한다.

        Returns:
            요약된 텍스트. 대화가 충분히 길지 않으면 SHORT_CONVERSATION_MESSAGE를,
            API 호출에 실패하면 사용자 친화적인 에러 메시지를 반환한다.
        """
        # 1. 대화가 너무 짧으면(MIN_MESSAGES_FOR_SUMMARY개 이하) 요약할 내용이
        #    부족하므로 바로 안내
        if len(self.messages) <= MIN_MESSAGES_FOR_SUMMARY:
            logger.debug(
                "요약 스킵: 메시지 수(%d) <= 기준(%d)",
                len(self.messages),
                MIN_MESSAGES_FOR_SUMMARY,
            )
            return SHORT_CONVERSATION_MESSAGE

        # 2. 요약 요청용 임시 메시지 - self.messages에는 반영하지 않는다
        summary_request: dict[str, Any] = {
            "role": "user",
            "content": SUMMARY_REQUEST_PROMPT,
        }
        temp_messages: list[dict[str, Any]] = self.messages + [summary_request]

        try:
            # 3. 임시 메시지 목록으로만 API 호출
            #    (일관된 요약을 위해 temperature를 낮게 설정)
            logger.debug("대화 요약 API 호출 (메시지 %d개)", len(temp_messages))
            response: ChatCompletion = self.client.chat.completions.create(
                model=self.model,
                messages=temp_messages,
                temperature=SUMMARY_TEMPERATURE,
                max_tokens=settings.MAX_TOKENS,
            )
        except AuthenticationError as error:
            # 요약 호출에서도 인증 실패는 별도로 명확히 안내한다.
            logger.error("대화 요약 중 OpenAI 인증 실패 - API 키 확인 필요: %s", error)
            return AUTH_ERROR_MESSAGE
        except OpenAIError as error:
            logger.error("대화 요약 API 호출 실패: %s", error)
            return (
                "죄송합니다, 지금 대화를 요약하는 중 문제가 발생했습니다. "
                "잠시 후 다시 시도해주세요."
            )

        # 4. 요약 텍스트 반환 (content가 없을 가능성에 대비해 빈 문자열로 방어)
        summary = response.choices[0].message.content or ""
        logger.info("대화 요약 완료 (원본 메시지 %d개)", len(self.messages))
        return summary

    def get_messages(self) -> list[dict[str, Any]]:
        """현재까지 누적된 대화 히스토리를 반환합니다.

        Returns:
            system/user/assistant/tool 역할(role)과 내용(content)을 담은
            딕셔너리의 리스트. 도구를 호출한 턴에는 tool_calls를 가진
            assistant 메시지와 그에 대응하는 tool 메시지가 포함된다.
        """
        return self.messages

    def get_turn_count(self) -> int:
        """성공적으로 완료된 대화 횟수를 반환합니다.

        Returns:
            사용자 질문-AI 응답 쌍이 성공적으로 완료된 횟수.
        """
        return self.turn_count

    def clear_history(self) -> None:
        """대화 히스토리를 초기화합니다.

        시스템 메시지(있는 경우)는 페르소나를 유지해야 하므로 남기고,
        그 뒤의 대화만 지운다. 검색 히스토리도 함께 비운다.
        """
        system_messages = [
            message for message in self.messages if message.get("role") == "system"
        ]
        self.messages = system_messages
        self.turn_count = 0
        self.state = STATE_IDLE

        # 검색 히스토리도 초기화
        if self.search_agent:
            self.search_agent.clear_history()

        logger.info("대화 히스토리 초기화됨")

    # ------------------------------------------------------------------
    # 검색 관련 메서드 (2주차 추가)
    # ------------------------------------------------------------------

    def get_last_search_sources(self) -> list[str]:
        """마지막 검색의 출처 목록을 반환합니다.

        Returns:
            list[str]: 출처 URL 목록. 검색 기록이 없으면 빈 리스트.
        """
        if self.search_agent:
            return self.search_agent.get_sources()
        return []

    def get_search_count(self) -> int:
        """총 검색 횟수를 반환합니다.

        Returns:
            int: 검색 횟수. 검색이 비활성화되어 있으면 0.
        """
        if self.search_agent:
            return self.search_agent.get_search_count()
        return 0

    def is_search_enabled(self) -> bool:
        """검색 기능 활성화 여부를 반환합니다.

        Returns:
            bool: 검색 기능 활성화 여부.
        """
        return self.enable_search

    def load_conversation(self, file_path: str | Path) -> None:
        """JSON 파일에서 대화 히스토리를 불러와 현재 대화를 대체합니다.

        두 가지 저장 형식을 모두 지원한다.
        - 1주차 형식: 메시지 딕셔너리의 리스트
        - 2주차 형식: {"messages": [...], "search_count": ...} 형태의 딕셔너리
          (main.py의 save_conversation()이 현재 저장하는 형식)

        Args:
            file_path: 불러올 JSON 파일 경로.

        Raises:
            ConversationLoadError: 파일을 읽을 수 없거나, JSON 형식이
                잘못되었거나, 메시지 구조(각 항목의 role/content)가
                예상과 다른 경우.
        """
        path = Path(file_path)

        try:
            raw_text = path.read_text(encoding="utf-8")
        except OSError as error:
            logger.error("대화 불러오기 실패 (%s): %s", path, error)
            raise ConversationLoadError(f"파일을 읽지 못했습니다: {error}") from error

        try:
            raw_data = json.loads(raw_text)
        except json.JSONDecodeError as error:
            logger.error("대화 불러오기 실패 - JSON 형식 오류 (%s): %s", path, error)
            raise ConversationLoadError(
                f"올바른 JSON 형식이 아닙니다: {error}"
            ) from error

        # 2주차 형식(딕셔너리)이면 messages 키를 꺼내고, 1주차 형식(리스트)이면
        # 그대로 사용한다.
        loaded = (
            raw_data.get("messages") if isinstance(raw_data, dict) else raw_data
        )

        # 각 항목이 {"role": ..., "content": ...} 형태의 딕셔너리인지 검증한다.
        is_valid = isinstance(loaded, list) and all(
            isinstance(item, dict) and "role" in item and "content" in item
            for item in loaded
        )
        if not is_valid:
            logger.error("대화 불러오기 실패 - 예상한 메시지 구조가 아님 (%s)", path)
            raise ConversationLoadError(
                "대화 파일 구조가 올바르지 않습니다. "
                '각 항목은 {"role": ..., "content": ...} 형태여야 합니다.'
            )

        self.messages = loaded
        self.turn_count = sum(1 for item in loaded if item.get("role") == "assistant")
        self.state = STATE_IDLE
        logger.info("대화 불러오기 완료 (%s, 메시지 %d개)", path, len(self.messages))
