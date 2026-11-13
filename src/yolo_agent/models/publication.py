"""人工模型发布记录与当前生产指针，Agent 无权写入。"""

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from yolo_agent.models.database import Base
from yolo_agent.models.dataset import new_id, utc_now


class ModelPublication(Base):
    """每次人工确认生成独立不可覆盖的发布审计和权重副本。"""

    __tablename__="model_publications"
    id: Mapped[str]=mapped_column(String(36),primary_key=True,default=new_id)
    project_id: Mapped[str]=mapped_column(ForeignKey("projects.id"),index=True)
    model_id: Mapped[str]=mapped_column(ForeignKey("model_registry.id"))
    previous_model_id: Mapped[str|None]=mapped_column(ForeignKey("model_registry.id"))
    evaluation_id: Mapped[str]=mapped_column(ForeignKey("evaluation_tasks.id"))
    reviewer: Mapped[str]
    reason: Mapped[str]
    weight_path: Mapped[str]
    weight_sha256: Mapped[str]
    manifest_path: Mapped[str]
    manifest_sha256: Mapped[str]
    created_at: Mapped[str]=mapped_column(default=utc_now)


class ProjectProductionModel(Base):
    """单项目只有一个当前生产版本，替换须显式确认旧指针。"""

    __tablename__="project_production_models"
    project_id: Mapped[str]=mapped_column(ForeignKey("projects.id"),primary_key=True)
    model_id: Mapped[str]=mapped_column(ForeignKey("model_registry.id"))
    publication_id: Mapped[str]=mapped_column(ForeignKey("model_publications.id"))
    updated_at: Mapped[str]=mapped_column(default=utc_now)
