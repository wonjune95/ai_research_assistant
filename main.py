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
from src.exceptions import ConversationSaveError

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# 종료를 트리거하는 명령어 모음 (대소문자 구분 없이 비교하므로 소문자로만 정의)
EXIT_COMMANDS: set[str] = {"quit", "exit", "종료"}
SAVE_COMMAND: str = "save"
SUMMARY_COMMAND: str = "summary"
CLEAR_COMMAND: str = "clear"
SOURCES_COMMAND: str = "sources"
STATUS_COMMAND: str = "status"

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
    print()
    print(SEPARATOR)
    print("🔍 AI 리서치 어시스턴트 v2.0")
    print("   웹 검색 기능이 추가되었습니다!")
    print(SEPARATOR)
    print()
    print("📌 사용 가능한 명령어:")
    print("  • quit / exit / 종료  : 프로그램 종료")
    print("  • save               : 대화 저장")
    print("  • clear              : 대화 히스토리 초기화")
    print("  • sources            : 마지막 검색 출처 보기")
    print("  • status             : 현재 상태 확인")
    print()
    print("💡 검색 활용 팁:")
    print("  • '~에 대해 조사해줘' → 웹 검색 실행")
    print("  • '최신 ~ 알려줘' → 최신 정보 검색")
    print("  • '~ 뉴스 찾아줘' → 관련 뉴스 검색")
    print()
    print(SEPARATOR)
    print()


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

    # 메시지 리스트만 저장하던 1주차 형식에서, 메타데이터를 함께 담는
    # 딕셔너리 형식으로 확장했다. load_conversation()은 두 형식을 모두 읽는다.
    save_data: dict = {
        "timestamp": datetime.now().isoformat(),
        "messages": manager.get_messages(),
        "turn_count": manager.get_turn_count(),
        "state": manager.state,
        # 검색 관련 정보
        "search_enabled": manager.is_search_enabled(),
        "search_count": manager.get_search_count(),
    }

    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        file_path.write_text(
            json.dumps(save_data, ensure_ascii=False, indent=2),
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


def handle_quit(manager: ConversationManager) -> bool:
    """종료 전 저장 여부를 사용자에게 확인하고, 원하면 저장합니다.

    Args:
        manager: 저장할 대화 히스토리를 가진 ConversationManager 인스턴스.

    Returns:
        bool: 항상 True (명령어를 처리했음을 의미).
    """
    save_choice = input("대화를 저장하시겠습니까? (y/n): ").strip().lower()
    if save_choice == CONFIRM_YES:
        _save_and_report(manager)
    return True


def handle_command(command: str, manager: ConversationManager) -> bool:
    """입력을 명령어로 해석해 처리합니다.

    Args:
        command: 사용자가 입력한 문자열.
        manager: 명령을 적용할 ConversationManager 인스턴스.

    Returns:
        bool: 명령어로 처리했으면 True, 명령어가 아니면 False.
            False면 호출부가 이를 일반 대화 입력으로 처리한다.
    """
    command = command.lower().strip()

    # 종료 명령어
    if command in EXIT_COMMANDS:
        return handle_quit(manager)

    # 저장 명령어
    if command == SAVE_COMMAND:
        _save_and_report(manager)
        return True

    # 요약 명령어
    if command == SUMMARY_COMMAND:
        print(SEPARATOR)
        print(manager.summarize_conversation())
        print(SEPARATOR)
        return True

    # 초기화 명령어
    if command == CLEAR_COMMAND:
        manager.clear_history()
        print("\n🧹 대화 히스토리를 초기화했습니다.\n")
        return True

    # 출처 보기 명령어
    if command == SOURCES_COMMAND:
        sources = manager.get_last_search_sources()
        if sources:
            print("\n📚 마지막 검색 출처:")
            for index, source in enumerate(sources, 1):
                print(f"  {index}. {source}")
            print()
        else:
            print("\n검색 기록이 없습니다.\n")
        return True

    # 상태 확인 명령어
    if command == STATUS_COMMAND:
        print("\n📊 현재 상태:")
        search_state = "활성화" if manager.is_search_enabled() else "비활성화"
        print(f"  • 검색 기능: {search_state}")
        print(f"  • 대화 횟수: {manager.get_turn_count()}회")
        print(f"  • 검색 횟수: {manager.get_search_count()}회")
        print()
        return True

    return False


def main() -> None:
    """AI 리서치 어시스턴트의 콘솔 진입점.

    ConversationManager를 초기화한 뒤, 사용자 입력을 받아 일반 대화 및
    handle_command()가 처리하는 명령어를 반복 실행합니다.

    Returns:
        None.
    """
    try:
        manager = ConversationManager(
            system_message=SYSTEM_PROMPT,
            enable_search=True,
        )

        # 검색 기능 상태 출력
        if manager.is_search_enabled():
            print("✅ 검색 기능이 활성화되었습니다.\n")
        else:
            print("⚠️ 검색 기능이 비활성화되었습니다. (API 키 확인 필요)\n")

    except Exception as error:
        # API 키가 없거나 잘못된 경우 등, 초기화 단계에서 발생하는 오류.
        # ConfigurationError는 원인을 이미 사용자 친화적인 메시지로 감싸서
        # 던지므로 그대로 출력한다.
        logger.error("초기화 실패: %s", error)
        print(f"❌ 초기화 실패: {error}")
        print("환경 설정을 확인해주세요. (.env 파일)")
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

            # 명령어면 handle_command가 처리한다. 종료 명령은 처리 후
            # 루프를 빠져나가야 하므로 여기서 별도로 판단한다.
            if handle_command(command, manager):
                if command in EXIT_COMMANDS:
                    break
                continue

            # 일반 대화 처리
            # 검색이 실행되면 응답까지 시간이 걸리므로 진행 중임을 알린다.
            print("\n🔄 처리 중...")
            reply: str = manager.chat(user_input)
            print(f"\nAI: {reply}\n")

    except KeyboardInterrupt:
        # Ctrl+C로 강제 종료한 경우
        print("\n\n사용자에 의해 중단되었습니다.")

    except Exception as error:
        # 예상치 못한 오류 발생 시 사용자에게 알리고 종료
        logger.exception("예상치 못한 오류 발생")
        print(f"\n예상치 못한 오류가 발생했습니다: {error}")

    else:
        print()
        print(SEPARATOR)
        print("👋 대화를 종료합니다. 안녕히 가세요!")
        print(f"   총 대화: {manager.get_turn_count()}회")
        print(f"   총 검색: {manager.get_search_count()}회")
        print(SEPARATOR)
        print()


if __name__ == "__main__":
    main()
