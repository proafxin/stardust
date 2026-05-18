from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_port: int
    postgres_db: str
    postgres_user: str
    postgres_password: str
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
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
RERANKER_MODEL = "mixedbread-ai/mxbai-rerank-base-v2"
ATOM_TOKEN_LIMIT = 512
LLM_BATCH_TOKEN_LIMIT = 12_000
NLP_BATCH_SIZE = 256
NLP_COMMIT_BATCH_SIZE = 512
EMBEDDING_BATCH_SIZE = 3072
EMBEDDING_INTERNAL_BATCH_SIZE = 256
RRF_K = 60
ENTITY_MERGE_THRESHOLD = 0.98
GLOBAL_MERGE_THRESHOLD = 0.96

settings = Settings()
