# .env에 있는 환경변수를 받아옴
from pydantic_settings import BaseSettings, SettingsConfigDict
import os, sys

ROOT_DIR = os.path.dirname(__file__)  # Project Directory
ENV_PATH = os.path.join(ROOT_DIR, ".env")
DATA_PATH = os.path.join(ROOT_DIR, "data")

class Settings(BaseSettings):
    openai_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str

    model_config = SettingsConfigDict(
        env_file=ENV_PATH,
        env_file_encoding="utf-8",
        from_attributes=True,
    )

settings = Settings()

