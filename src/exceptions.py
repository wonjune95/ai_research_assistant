"""AI 리서치 어시스턴트 전용 커스텀 예외 클래스 모음.

호출부(main.py 등)가 openai SDK나 파일 시스템의 구체적인 예외 타입을
직접 알 필요 없이, 이 모듈이 정의하는 의미 있는 예외만 처리하면 되도록
하기 위해 사용한다.
"""


class ResearchAssistantError(Exception):
    """이 프로젝트에서 발생하는 모든 커스텀 예외의 최상위 클래스."""


class ConfigurationError(ResearchAssistantError):
    """API 키 등 필수 설정이 누락되었거나 잘못되었을 때 발생한다.

    OPENAI_API_KEY가 없거나 OpenAI 클라이언트를 생성할 수 없는 경우에
    사용한다.
    """


class ConversationSaveError(ResearchAssistantError):
    """대화 히스토리를 파일로 저장하는 데 실패했을 때 발생한다.

    디스크 공간 부족, 권한 문제 등 파일 I/O 오류를 감싸는 데 사용한다.
    """


class ConversationLoadError(ResearchAssistantError):
    """저장된 대화 히스토리를 파일에서 불러오는 데 실패했을 때 발생한다.

    파일을 읽을 수 없거나, JSON 형식이 잘못되었거나, 메시지 구조
    (role/content)가 예상과 다른 경우에 사용한다.
    """
