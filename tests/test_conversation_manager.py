"""ConversationManager에 대한 기본 단위 테스트.

OpenAI API를 실제로 호출하지 않아도 통과하도록 작성되었다.
- ConversationManager 초기화는 OpenAI 클라이언트 "객체"만 생성할 뿐 네트워크
  호출을 하지 않으므로, 더미 API 키만 있으면 실제 키 없이도 안전하다.
- chat()처럼 실제로 API를 호출하는 메서드는 client.chat.completions.create를
  unittest.mock으로 대체해 네트워크 호출 없이 동작을 검증한다.
"""

from unittest.mock import MagicMock

import pytest

from config import settings
from src.conversation_manager import (
    STATE_IDLE,
    STATE_RESEARCHING,
    STATE_RESPONDING,
    ConversationManager,
)
from src.exceptions import ConfigurationError

# 테스트 전용 더미 API 키. 실제 OpenAI에 호출되지 않으므로 유효한 키일 필요는 없다.
DUMMY_API_KEY = "sk-test-dummy-key-for-unit-tests"
TEST_SYSTEM_MESSAGE = "당신은 테스트용 시스템 메시지입니다."


@pytest.fixture(autouse=True)
def dummy_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """모든 테스트에서 OPENAI_API_KEY를 더미 값으로 설정한다.

    ConversationManager.__init__은 키가 존재하면 OpenAI 클라이언트 객체를
    생성하기만 할 뿐 네트워크 호출은 하지 않으므로, 실제 API 키 없이도
    안전하게 초기화 테스트를 진행할 수 있다.
    """
    monkeypatch.setenv("OPENAI_API_KEY", DUMMY_API_KEY)


class _FakeMessage:
    """OpenAI 응답의 message 객체를 흉내내는 더미 객체."""

    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    """OpenAI 응답의 choice 객체를 흉내내는 더미 객체."""

    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeChatCompletion:
    """client.chat.completions.create()의 반환값을 흉내내는 더미 객체."""

    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class TestConversationManagerInit:
    """1. ConversationManager 초기화 테스트."""

    def test_init_starts_with_empty_history_and_idle_state(self) -> None:
        manager = ConversationManager()

        assert manager.messages == []
        assert manager.turn_count == 0
        assert manager.state == STATE_IDLE

    def test_init_uses_default_model_from_settings(self) -> None:
        manager = ConversationManager()

        assert manager.model == settings.DEFAULT_MODEL

    def test_init_accepts_custom_model(self) -> None:
        manager = ConversationManager(model="gpt-4o")

        assert manager.model == "gpt-4o"

    def test_init_without_api_key_raises_configuration_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "")

        with pytest.raises(ConfigurationError):
            ConversationManager()


class TestSystemMessage:
    """2. System Message 설정 테스트."""

    def test_system_message_added_as_first_message(self) -> None:
        manager = ConversationManager(system_message=TEST_SYSTEM_MESSAGE)

        assert len(manager.messages) == 1
        assert manager.messages[0] == {
            "role": "system",
            "content": TEST_SYSTEM_MESSAGE,
        }

    def test_no_system_message_means_empty_history(self) -> None:
        manager = ConversationManager(system_message=None)

        assert manager.messages == []

    def test_empty_string_system_message_is_not_added(self) -> None:
        # "" 는 falsy이므로 system_message가 있다고 보지 않는다
        # (if system_message: 분기).
        manager = ConversationManager(system_message="")

        assert manager.messages == []


class TestMessageAppending:
    """3. 메시지 추가 테스트 (실제 API 호출은 mock으로 대체)."""

    def test_chat_appends_user_and_assistant_messages(self) -> None:
        manager = ConversationManager(system_message=TEST_SYSTEM_MESSAGE)
        manager.client.chat.completions.create = MagicMock(
            return_value=_FakeChatCompletion("안녕하세요, 무엇을 도와드릴까요?")
        )

        reply = manager.chat("안녕")

        assert reply == "안녕하세요, 무엇을 도와드릴까요?"
        # system(1) + user(1) + assistant(1) = 3개
        assert len(manager.messages) == 3
        assert manager.messages[1] == {"role": "user", "content": "안녕"}
        assert manager.messages[2] == {"role": "assistant", "content": reply}
        assert manager.turn_count == 1
        assert manager.state == STATE_IDLE

    def test_chat_increments_turn_count_across_multiple_calls(self) -> None:
        manager = ConversationManager()
        manager.client.chat.completions.create = MagicMock(
            return_value=_FakeChatCompletion("응답")
        )

        manager.chat("첫 번째 질문")
        manager.chat("두 번째 질문")

        assert manager.turn_count == 2
        assert len(manager.messages) == 4  # user/assistant 쌍이 2번 추가됨


class TestDetermineState:
    """4. 상태 판단(determine_state) 테스트."""

    @pytest.fixture
    def manager(self) -> ConversationManager:
        return ConversationManager()

    @pytest.mark.parametrize(
        "user_input",
        [
            "시장을 조사해줘",
            "경쟁사 분석 좀 해줘",
            "이 논문 리서치 부탁해",
            "최신 트렌드 알아봐줘",
            "맛집 찾아봐",
        ],
    )
    def test_research_keywords_trigger_researching_state(
        self, manager: ConversationManager, user_input: str
    ) -> None:
        assert manager.determine_state(user_input) == STATE_RESEARCHING

    @pytest.mark.parametrize(
        "user_input",
        ["안녕", "오늘 날씨 어때?", "고마워!"],
    )
    def test_non_research_input_triggers_responding_state(
        self, manager: ConversationManager, user_input: str
    ) -> None:
        assert manager.determine_state(user_input) == STATE_RESPONDING

    def test_empty_input_defaults_to_responding_state(
        self, manager: ConversationManager
    ) -> None:
        assert manager.determine_state("") == STATE_RESPONDING
