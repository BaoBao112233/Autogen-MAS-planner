from functools import lru_cache
import os
from pydantic_settings import BaseSettings, SettingsConfigDict


@lru_cache
def get_env_filename():
    runtime_env = os.getenv("ENV")
    return f".env.{runtime_env}" if runtime_env else ".env"


class EnvironmentSettings(BaseSettings):
    # Application settings
    API_VERSION: str
    APP_NAME: str
    APP_DESC: str
    APP_PORT: int = 9000
    # Vertex AI settings
    MODEL_NAME: str = "gemini-2.5-pro"
    GOOGLE_CLOUD_PROJECT: str
    GOOGLE_CLOUD_LOCATION: str = "us-east1"
    GOOGLE_APPLICATION_CREDENTIALS: str = "service-account.json"
    # Chatbot settings
    CHATBOTS_URL: str
    CHATBOTS_INFO_URL: str
    # CRM settings
    CRM_API_URL: str
    CRM_AUTH_TOKEN: str 
    # Memory settings
    MEMORY_API_URL: str
    MEMORY_AUTH_TOKEN: str
    # Database settings
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_DB: int
    TTL_SECONDS: int = 3600
    PLAN_API_BASE_URL: str
    PLAN_API_KEY: str
    OXII_ROOT_API_URL: str
    # ElevenLabs API settings
    ELEVENLABS_BASE_URL: str
    ELEVENLABS_API_KEY: str
    ELEVENLABS_MODEL_ID: str
    ELEVENLABS_VOICE_ID: str
    ELEVENLABS_STABILITY: float = 0.5
    ELEVENLABS_SIMILARITY_BOOST: float = 0.5
    # Langfuse settings
    LANGFUSE_PUBLIC_KEY: str
    LANGFUSE_SECRET_KEY: str
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    LANGFUSE_PROJECT_NAME: str = "MAS-Planning"
    # Debug settings
    MCP_SERVER_URL: str
    MAX_TURNS: int = 20
    LIMIT_MINUTES: int = 10
    MAX_MSG: int = 12
    DEBUG_MODE: bool = False

    model_config = SettingsConfigDict(env_file=get_env_filename(), env_file_encoding="utf-8")


@lru_cache
def get_environment_variables():
    env_settings = EnvironmentSettings()
    # Set the Google Application Credentials environment variable
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = env_settings.GOOGLE_APPLICATION_CREDENTIALS
    return env_settings

env = get_environment_variables()