"""OpenAI Chat Completions API를 이용해 대화 히스토리를 관리하는 모듈."""

import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AuthenticationError, OpenAI, OpenAIError
from openai.types.chat import ChatCompletion

from config import settings
from src.exceptions import ConfigurationError, ConversationLoadError

load_dotenv()

logger = logging.getLogger(__name__)

# ConversationManager 기본 설정값 (모델명/temperature/재시도 횟수는
# config.settings에서 가져온다. 요약 전용 temperature는 이 모듈에서만
# 쓰이는 값이라 로컬 상수로 둔다.)
SUMMARY_TEMPERATURE = 0.3

# determine_state / chat이 사용하는 대화 상태 상수
STATE_IDLE = "idle"
STATE_RESPONDING = "responding"
STATE_RESEARCHING = "researching"

# summarize_conversation이 사용하는 상수
MIN_MESSAGES_FOR_SUMMARY = 3
SHORT_CONVERSATION_MESSAGE = "대화가 충분히 길지 않습니다"
SUMMARY_REQUEST_PROMPT = "지금까지의 대화를 3문장으로 요약해주세요."

# chat()과 summarize_conversation() 양쪽에서 공통으로 쓰는 인증 실패 안내 메시지.
# 재시도해도 해결되지 않는 오류이므로 문구를 한 곳에서만 관리한다.
AUTH_ERROR_MESSAGE = (
    "OpenAI API 키가 유효하지 않습니다. .env 파일의 OPENAI_API_KEY 값을 확인해주세요."
)


class ConversationManager:
    """OpenAI Chat Completions API를 이용해 대화 히스토리를 관리하는 클래스.

    Attributes:
        client: OpenAI API 호출에 사용하는 클라이언트.
        model: 사용할 OpenAI 모델 이름.
        messages: 지금까지 주고받은 대화 히스토리 (system/user/assistant 메시지).
        turn_count: 성공적으로 완료된 대화(사용자 질문-AI 응답 쌍)의 횟수.
        state: 현재 대화 상태 (STATE_IDLE | STATE_RESPONDING | STATE_RESEARCHING).
    """

    # determine_state에서 사용하는 리서치 모드 감지 키워드.
    # NOTE: 4주차에 LLM 기반 판단(의도 분류)으로 고도화 예정. 현재는 단순
    # 키워드 포함 여부로만 판단하는 1주차 수준의 규칙 기반 구현이다.
    RESEARCH_KEYWORDS: list[str] = ["조사", "분석", "리서치", "알아봐", "찾아봐"]

    def __init__(
        self,
        system_message: str | None = None,
        model: str = settings.DEFAULT_MODEL,
    ) -> None:
        """ConversationManager를 초기화합니다.

        Args:
            system_message: 대화의 시스템 프롬프트. 주어지면 messages의
                첫 항목으로 추가됩니다.
            model: 사용할 OpenAI 모델 이름. 기본값은 config.settings.DEFAULT_MODEL.

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
        self.messages: list[dict[str, str]] = []
        self.turn_count: int = 0
        self.state: str = STATE_IDLE

        if system_message:
            self.messages.append({"role": "system", "content": system_message})

        logger.info("ConversationManager 초기화 완료 (model=%s)", model)

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

    def _call_with_retry(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_retries: int,
    ) -> tuple[ChatCompletion | None, str | None]:
        """OpenAI API를 호출하고, 실패 시 지수 백오프로 재시도합니다.

        attempt은 0부터 시작하며 "이번이 몇 번째 시도인지"를 의미한다.
        - attempt 0: 최초 시도 (재시도 아님, 대기 없음)
        - attempt 1~max_retries: 재시도. 대기 시간은 2 ** attempt 초
          (1차 재시도 attempt=1 -> 2초, 2차 attempt=2 -> 4초, 3차 attempt=3 -> 8초)

        인증 실패(AuthenticationError)는 재시도해도 결과가 바뀌지 않으므로
        백오프 없이 즉시 중단한다.

        Args:
            messages: API에 전달할 메시지 목록.
            temperature: 응답 생성에 사용할 temperature.
            max_retries: 재시도할 최대 횟수.

        Returns:
            성공하면 (ChatCompletion, None), 실패하면
            (None, 사용자 친화적인 에러 메시지) 튜플.
        """
        last_error: OpenAIError | None = None
        for attempt in range(max_retries + 1):
            try:
                logger.debug("OpenAI API 호출 시도 %d/%d", attempt + 1, max_retries + 1)
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=settings.MAX_TOKENS,
                )
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

    def chat(self, user_input: str, max_retries: int = settings.MAX_RETRIES) -> str:
        """사용자 입력을 히스토리에 추가하고 모델 응답을 반환합니다.

        API 호출이 일시적으로 실패하면 지수 백오프(exponential backoff)를
        적용해 최대 max_retries번까지 재시도한다 (대기 시간: 2초 -> 4초 -> 8초).

        Args:
            user_input: 사용자가 입력한 메시지.
            max_retries: API 호출 실패 시 재시도할 최대 횟수.
                기본값 config.settings.MAX_RETRIES.

        Returns:
            모델이 생성한 응답 텍스트. 재시도를 모두 소진하고도 실패하면
            사용자 친화적인 에러 메시지를 반환합니다.
        """
        # 1. 사용자 입력을 바탕으로 상태를 판단하고 갱신
        #    (idle -> responding/researching)
        self.state = self.determine_state(user_input)

        # 2. 사용자 입력을 히스토리에 추가
        self.messages.append({"role": "user", "content": user_input})
        logger.debug(
            "사용자 입력 히스토리에 추가 (state=%s, 현재 메시지 수=%d)",
            self.state,
            len(self.messages),
        )

        # 3. 전체 히스토리를 담아 API 호출 (실패 시 내부적으로 재시도)
        response, error_message = self._call_with_retry(
            self.messages, settings.DEFAULT_TEMPERATURE, max_retries
        )

        if error_message is not None:
            # 응답을 받지 못했으므로 방금 추가한 사용자 메시지를 히스토리에서
            # 되돌려 상태를 일관되게 유지한다.
            self.messages.pop()
            self.state = STATE_IDLE
            return error_message

        # 4. AI 응답을 히스토리에 추가 (content가 없을 가능성에 대비해 빈 문자열로 방어)
        assert response is not None
        reply: str = response.choices[0].message.content or ""
        self.messages.append({"role": "assistant", "content": reply})
        self.turn_count += 1

        # 5. 응답 처리가 끝났으므로 다시 대기 상태로 전환
        self.state = STATE_IDLE

        # 6. AI 응답 텍스트 반환
        logger.info("대화 응답 수신 완료 (turn_count=%d)", self.turn_count)
        return reply

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
        summary_request: dict[str, str] = {
            "role": "user",
            "content": SUMMARY_REQUEST_PROMPT,
        }
        temp_messages: list[dict[str, str]] = self.messages + [summary_request]

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

    def get_messages(self) -> list[dict[str, str]]:
        """현재까지 누적된 대화 히스토리를 반환합니다.

        Returns:
            system/user/assistant 역할(role)과 내용(content)을 담은
            딕셔너리의 리스트.
        """
        return self.messages

    def get_turn_count(self) -> int:
        """성공적으로 완료된 대화 횟수를 반환합니다.

        Returns:
            사용자 질문-AI 응답 쌍이 성공적으로 완료된 횟수.
        """
        return self.turn_count

    def load_conversation(self, file_path: str | Path) -> None:
        """JSON 파일에서 대화 히스토리를 불러와 현재 대화를 대체합니다.

        main.py의 save_conversation()이 저장하는 형식, 즉 각 항목이
        {"role": ..., "content": ...} 인 딕셔너리의 리스트와 호환된다.

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
            loaded = json.loads(raw_text)
        except json.JSONDecodeError as error:
            logger.error("대화 불러오기 실패 - JSON 형식 오류 (%s): %s", path, error)
            raise ConversationLoadError(
                f"올바른 JSON 형식이 아닙니다: {error}"
            ) from error

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
