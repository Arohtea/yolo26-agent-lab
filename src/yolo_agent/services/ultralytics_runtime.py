"""在项目隔离目录中配置成熟训练库，避免隐式安装与远程追踪。"""

import os

from yolo_agent.settings import Settings


def configure_ultralytics(settings: Settings) -> None:
    """导入训练库前设置离线策略和项目配置路径。

    :param settings: 平台持久化目录与预训练模型目录。
    :returns: 无。
    """
    os.environ["YOLO_CONFIG_DIR"] = str((settings.storage_dir / "ultralytics").resolve())
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["MPLBACKEND"] = "Agg"
    from ultralytics import settings as ultra_settings

    ultra_settings.update({"sync":False,"clearml":False,"comet":False,"dvc":False,"raytune":False,"mlflow":False,"wandb":False,"tensorboard":False,"datasets_dir":str((settings.storage_dir / "datasets").resolve()),"runs_dir":str((settings.storage_dir / "training").resolve()),"weights_dir":str(settings.training_model_dir.resolve())})
