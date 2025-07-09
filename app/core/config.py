from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # 루트 디렉토리

class Settings(BaseSettings):
    DB_HOST: str
    DB_USER: str
    DB_PASSWORD: str
    DB_NAME: str

    GEMINI_API_KEY: str
    OPENAI_API_KEY: str

    CACHE_PATH_STR: str = Field(..., alias="CACHE_PATH")
    INDEX_PATH_STR: str = Field(..., alias="INDEX_PATH")
    CSV_PATH_STR: str = Field(..., alias="CSV_PATH")

    @property
    def CACHE_PATH(self) -> Path:
        return BASE_DIR / self.CACHE_PATH_STR
    
    @property
    def INDEX_PATH(self) -> Path:
        return BASE_DIR / self.INDEX_PATH_STR
    
    @property
    def CSV_PATH(self) -> Path:
        return BASE_DIR / self.CSV_PATH_STR

    class Config:
        env_file = BASE_DIR / ".env"

settings = Settings()