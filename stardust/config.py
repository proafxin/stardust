from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_port: int
    postgres_db: str
    postgres_user: str
    postgres_password: str
    ollama_host: str = "localhost"
    ollama_port: int = 11434

    @property
    def postgres_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@localhost:{self.postgres_port}/{self.postgres_db}?ssl=disable"
        )


COREF_MODEL = "biu-nlp/f-coref"
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
RERANKER_MODEL = "mixedbread-ai/mxbai-rerank-base-v2"
RRF_K = 60

settings = Settings()
