from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_port: int
    postgres_db: str
    postgres_user: str
    postgres_password: str
    groq_api_key: str
    gemini_api_key: str
    redis_port: int = 6379
    redis_host: str = "localhost"
    ollama_host: str = "localhost"
    ollama_port: int = 11434

    @property
    def postgres_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@localhost:{self.postgres_port}/{self.postgres_db}?ssl=disable"
        )


SPACY_MODEL = "en_core_web_trf"
EMBEDDING_MODEL = "microsoft/harrier-oss-v1-0.6b"
RERANKER_MODEL = "mixedbread-ai/mxbai-rerank-base-v2"
ATOM_TOKEN_LIMIT = 512
LLM_BATCH_TOKEN_LIMIT = 12_000
NLP_BATCH_SIZE = 2048
EMBEDDING_BATCH_SIZE = 2048
RRF_K = 60
ENTITY_MERGE_THRESHOLD = 0.98
GLOBAL_MERGE_THRESHOLD = 0.92

settings = Settings()
