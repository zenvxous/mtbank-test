from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    WHISPER_MODEL_SIZE: str = "medium"
    WHISPER_DEVICE: str = "cpu"
    WHISPER_COMPUTE_TYPE: str = "int8"
    WHISPER_DOWNLOAD_ROOT: str | None = None
    ASR_LANGUAGE: str = "ru"
    ASR_BEAM_SIZE: int = 5

    DIARIZATION_ENABLED: bool = True
    DIARIZATION_MODEL: str = "pyannote/speaker-diarization-3.1"
    HF_TOKEN: str | None = None

    MAX_UPLOAD_BYTES: int = 100 * 1024 * 1024

    OPENROUTER_API_KEY: str | None = None
    OPENROUTER_MODEL: str | None = None

    APP_NAME: str = "mtbank-call-analyer"
    LOG_LEVEL: str = "INFO"
    DEBUG: bool = False
    APP_ENV: str = "prod"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

settings = Settings()
