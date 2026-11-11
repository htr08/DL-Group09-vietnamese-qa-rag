from pydantic_settings import BaseSettings, SettingsConfigDict

# Lớp cấu hình ứng dụng, tự động tải từ biến môi trường và file .env
class Settings(BaseSettings):
    PROJECT_NAME: str = "Marketing_RAG"

    # Cơ sở dữ liệu PostgreSQL (bắt buộc)
    DATABASE_URL: str

    # Lưu trữ file (MinIO)
    MINIO_ENDPOINT: str
    MINIO_ROOT_USER: str
    MINIO_ROOT_PASSWORD: str
    MINIO_BUCKET_NAME: str = "marketing-docs"

    # AI và CSDL vector
    GEMINI_API_KEY: str
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = "cellphones_marketing_docs"

    # Hàng đợi tác vụ (Celery + Redis)
    REDIS_URL: str = "redis://localhost:6379/0"

    # Xác thực JWT
    JWT_SECRET_KEY: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # Cấu hình Pydantic: đọc từ file .env
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

# Instance cấu hình toàn cục (khởi tạo khi import file này)
settings = Settings()