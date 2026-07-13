from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration in one typed, env-overridable place."""
    model_config = SettingsConfigDict(env_prefix="LLM_", env_file=".env")

    app_name: str = "llm-inference-server"
    version: str = "0.1.0"

    host: str = "0.0.0.0"
    port: int = 8000

    # Stand-in model for CPU dev. Swap to the real H200 model later — just this line.
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    log_level: str = "INFO"


settings = Settings()