"""集中管理运行配置及持久化路径。"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, computed_field, model_validator
from pydantic_settings import (
    BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict, YamlConfigSettingsSource,
)


class Settings(BaseSettings):
    """配置优先级为显式参数、环境变量、.env、YAML、字段默认值。"""

    model_config = SettingsConfigDict(env_prefix="YOLO_", env_file=".env", extra="ignore")
    device: str = Field(default="auto", pattern=r"^(auto|cpu|cuda(?::\d+)?)$")
    storage_dir: Path = Path("storage")
    redis_url: str = "redis://127.0.0.1:6379/0"
    mlflow_tracking_uri: str = "http://127.0.0.1:5001"
    mlflow_public_url: str | None = None
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    queue_name: str = "yolo-compute"
    job_timeout: int = Field(default=180, ge=10)
    job_result_ttl: int = Field(default=86400, ge=60)
    runtime_report_interval: int = Field(default=30, ge=5)
    runtime_report_ttl: int = Field(default=120, ge=15)
    sqlite_busy_timeout_ms: int = Field(default=5000, ge=100)
    health_timeout: float = Field(default=3, gt=0, le=30)
    log_level: str = "INFO"
    dataset_job_timeout: int = Field(default=1800, ge=30)
    max_upload_files: int = Field(default=500, ge=1, le=1000)
    max_image_pixels: int = Field(default=40_000_000, ge=1024)
    max_dataset_images: int = Field(default=20_000, ge=3)
    max_similarity_pairs: int = Field(default=100_000, ge=1)
    annotation_model_path: Path = Path("storage/pretrained/grounding-dino-tiny")
    annotation_job_timeout: int = Field(default=7200, ge=60)
    cvat_url: str = "http://localhost:8080"
    cvat_public_url: str | None = None
    cvat_username: str = ""
    cvat_password: SecretStr = SecretStr("")
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = ""
    llm_api_key: SecretStr = SecretStr("")
    llm_extra_body: dict = Field(default_factory=dict)
    training_config_path: Path = Path("configs/training.yaml")
    training_model_dir: Path = Path("storage/pretrained/yolo26")
    training_job_timeout: int = Field(default=172800, ge=60)
    evaluation_job_timeout: int = Field(default=7200, ge=60)
    export_job_timeout: int = Field(default=1800, ge=60)
    agent_config_path: Path = Path("configs/agent.yaml")
    agent_stagnation_rounds: int | None = Field(default=None,ge=1,le=20)
    agent_minimum_improvement: float | None = Field(default=None,ge=0,le=0.1)
    optimization_config_path: Path = Path("configs/optimization.yaml")
    agent_poll_seconds: float = Field(default=3,ge=1,le=60)
    agent_queue_wait_seconds: int = Field(default=300,ge=30,le=3600)
    # 多张指标图的视觉推理明显慢于纯文本调用，上限放宽到 5 分钟。
    llm_timeout_seconds: float = Field(default=90,ge=5,le=300)
    # 部分兼容服务不支持 required，可改为 auto，模型仍须每轮调用工具。
    llm_tool_choice: Literal["required","auto"] = "required"
    max_gpu_memory_ratio: float = Field(default=0.9, gt=0.1, le=0.9)

    @classmethod
    def settings_customise_sources(
        cls, settings_cls: type[BaseSettings], init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource, dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """按部署覆盖顺序组合配置源。

        :param settings_cls: 当前配置类型。
        :param init_settings: 显式构造参数源。
        :param env_settings: 进程环境变量源。
        :param dotenv_settings: 本地 .env 源。
        :param file_secret_settings: 文件密钥源。
        :returns: 从高到低排序的配置源。
        """
        config_path = Path(os.environ.get("YOLO_CONFIG", "configs/app.yaml"))
        if "YOLO_CONFIG" in os.environ and not config_path.is_file():
            raise ValueError(f"指定配置文件不存在：{config_path}")
        return (
            init_settings, env_settings, dotenv_settings, file_secret_settings,
            YamlConfigSettingsSource(settings_cls, yaml_file=config_path),
        )

    @model_validator(mode="after")
    def validate_runtime_ttl(self) -> "Settings":
        """保证报告过期窗口足以覆盖两次心跳。

        :returns: 已校验的配置实例。
        """
        if self.runtime_report_ttl < self.runtime_report_interval * 2:
            raise ValueError("runtime_report_ttl 必须至少为报告周期的两倍")
        return self

    @computed_field
    @property
    def database_url(self) -> str:
        """业务库使用绝对路径，避免工作目录变化后连接到另一数据库。

        :returns: SQLite SQLAlchemy 连接地址。
        """
        return f"sqlite:///{self.storage_dir.resolve() / 'app.db'}"

    def prepare_storage(self) -> None:
        """创建当前阶段需要的持久化目录。

        :returns: 无。
        """
        for name in ("logs", "diagnostics", "imports", "images", "previews", "reports", "versions", "annotations", "exports", "pretrained", "training", "registry", "model_exports"):
            (self.storage_dir / name).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """复用进程内配置快照，避免每次请求读取配置文件。

    :returns: 当前进程的配置。
    """
    return Settings()
