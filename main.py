"""AI 리서치 어시스턴트 - 콘솔 대화 루프.

실행 예시:
    $ python main.py
    ============================================================
    AI 리서치 어시스턴트에 오신 것을 환영합니다!
    ============================================================
    궁금한 내용을 자유롭게 입력해주세요.

    사용 가능한 명령어
      save     : 지금까지의 대화를 파일로 저장합니다.
      summary  : 지금까지의 대화를 3문장으로 요약합니다.
      quit / exit / 종료 : 프로그램을 종료합니다.
    ============================================================

    You: 파이썬 리스트와 튜플의 차이가 뭐야?
    ============================================================
    AI: 리스트는 변경 가능(mutable)하고, 튜플은 변경 불가능(immutable)합니다...
    ============================================================

    You: 종료
    대화를 저장하시겠습니까? (y/n): y
    대화가 data/conversation_20260715_213000.json 에 저장되었습니다.
    총 1번 대화했습니다. 이용해주셔서 감사합니다.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from config import settings
from config.prompts import SYSTEM_PROMPT
from src.conversation_manager import ConversationManager
from src.exceptions import ConfigurationError, ConversationSaveError

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# 종료를 트리거하는 명령어 모음 (대소문자 구분 없이 비교하므로 소문자로만 정의)
EXIT_COMMANDS: set[str] = {"quit", "exit", "종료"}
SAVE_COMMAND: str = "save"
SUMMARY_COMMAND: str = "summary"

# 대화 구분선
SEPARATOR: str = "=" * 60

# 대화 저장 파일이 위치할 디렉터리 (디렉터리명은 config.settings.DATA_DIR에서 가져옴)
DATA_DIR: Path = Path(__file__).resolve().parent / settings.DATA_DIR

# 저장 파일명에 사용할 타임스탬프 포맷 (예: 20260715_213000)
SAVE_TIMESTAMP_FORMAT: str = settings.SAVE_FORMAT

# 종료 시 저장 여부를 묻는 프롬프트에서 "저장함"을 의미하는 응답
CONFIRM_YES: str = "y"


def print_welcome() -> None:
    """환영 메시지와 사용 가능한 명령어 안내를 화면에 출력합니다.

    Returns:
        None. 표준 출력으로 직접 메시지를 출력합니다.
    """
    print(SEPARATOR)
    print("AI 리서치 어시스턴트에 오신 것을 환영합니다!")
    print(SEPARATOR)
    print("궁금한 내용을 자유롭게 입력해주세요.\n")
    print("사용 가능한 명령어")
    print("  save     : 지금까지의 대화를 파일로 저장합니다.")
    print("  summary  : 지금까지의 대화를 3문장으로 요약합니다.")
    print("  quit / exit / 종료 : 프로그램을 종료합니다.")
    print(SEPARATOR)


def save_conversation(manager: ConversationManager) -> Path:
    """현재 대화 히스토리를 JSON 파일로 저장합니다.

    파일명은 저장 시점의 타임스탬프를 포함해 매번 고유하게 생성되므로,
    기존에 저장된 대화 파일을 덮어쓰지 않는다.

    Args:
        manager: 저장할 대화 히스토리를 가진 ConversationManager 인스턴스.

    Returns:
        저장된 JSON 파일의 경로.

    Raises:
        ConversationSaveError: 디렉터리 생성 또는 파일 쓰기 중 파일 I/O
            오류(권한 부족, 디스크 공간 부족 등)가 발생한 경우.
    """
    timestamp: str = datetime.now().strftime(SAVE_TIMESTAMP_FORMAT)
    file_path: Path = DATA_DIR / f"conversation_{timestamp}.json"

    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        file_path.write_text(
            json.dumps(manager.get_messages(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as error:
        # 디스크 공간 부족, 권한 거부 등 파일 I/O 오류를 앱 전용 예외로 감싸서
        # 호출부가 openai/파일 시스템 예외를 몰라도 되도록 한다.
        logger.error("대화 저장 실패 (%s): %s", file_path, error)
        raise ConversationSaveError(f"대화를 저장하지 못했습니다: {error}") from error

    logger.info("대화 저장 완료: %s", file_path)
    return file_path


def _save_and_report(manager: ConversationManager) -> None:
    """대화를 저장하고 결과를 화면에 안내합니다.

    저장 실패도 여기서 처리하므로, 호출부는 try/except 없이 호출만 하면 된다.
    ('save' 명령과 종료 시 저장 확인, 두 곳에서 공통으로 사용)

    Args:
        manager: 저장할 대화 히스토리를 가진 ConversationManager 인스턴스.
    """
    try:
        file_path = save_conversation(manager)
        print(f"대화가 {file_path}에 저장되었습니다.")
    except ConversationSaveError as error:
        logger.error("대화 저장 실패: %s", error)
        print(f"[저장 오류] {error}")


def _confirm_and_save_on_exit(manager: ConversationManager) -> None:
    """종료 전 저장 여부를 사용자에게 확인하고, 원하면 저장합니다.

    Args:
        manager: 저장할 대화 히스토리를 가진 ConversationManager 인스턴스.
    """
    save_choice = input("대화를 저장하시겠습니까? (y/n): ").strip().lower()
    if save_choice == CONFIRM_YES:
        _save_and_report(manager)


def main() -> None:
    """AI 리서치 어시스턴트의 콘솔 진입점.

    ConversationManager를 초기화한 뒤, 사용자 입력을 받아 일반 대화 및
    save/summary/종료 명령어를 처리하는 반복 루프를 실행합니다.

    Returns:
        None.
    """
    try:
        manager = ConversationManager(system_message=SYSTEM_PROMPT)
    except ConfigurationError as error:
        # API 키가 없거나 잘못된 경우 등, 초기화 단계에서 발생하는 설정 오류.
        # ConversationManager가 원인을 이미 사용자 친화적인 메시지로 감싸서
        # 던지므로 여기서는 그대로 출력한다.
        logger.error("ConversationManager 초기화 실패: %s", error)
        print(f"[설정 오류] {error}")
        return

    logger.info("ConversationManager 초기화 성공")
    print_welcome()

    try:
        while True:
            user_input: str = input("\nYou: ").strip()

            # 빈 입력은 무시하고 다시 입력받기
            if not user_input:
                continue

            # 명령어 비교는 대소문자를 구분하지 않도록 소문자로 변환해 사용
            command: str = user_input.lower()
            logger.debug("사용자 입력 수신: %r (command=%r)", user_input, command)

            # 종료 명령어 처리: 저장 여부를 확인한다 (저장 실패해도 종료는 계속 진행)
            if command in EXIT_COMMANDS:
                _confirm_and_save_on_exit(manager)
                break

            # 'save' 명령어: 즉시 현재 대화 저장
            if command == SAVE_COMMAND:
                _save_and_report(manager)
                continue

            # 'summary' 명령어: 지금까지의 대화 요약 출력
            if command == SUMMARY_COMMAND:
                print(SEPARATOR)
                print(manager.summarize_conversation())
                print(SEPARATOR)
                continue

            # 일반 대화 처리
            reply: str = manager.chat(user_input)
            print(SEPARATOR)
            print(f"AI: {reply}")
            print(SEPARATOR)

    except KeyboardInterrupt:
        # Ctrl+C로 강제 종료한 경우
        print("\n\n사용자에 의해 중단되었습니다.")

    except Exception as error:
        # 예상치 못한 오류 발생 시 사용자에게 알리고 종료
        logger.exception("예상치 못한 오류 발생")
        print(f"\n예상치 못한 오류가 발생했습니다: {error}")

    else:
        turn_count = manager.get_turn_count()
        print(f"\n총 {turn_count}번 대화했습니다. 이용해주셔서 감사합니다.")


if __name__ == "__main__":
    main()
