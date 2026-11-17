"""把注册权重转换为部署格式，并用对应运行时实际加载推理一次。"""

import hashlib
import logging
import shutil
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from rq import get_current_job
from sqlalchemy import update
from sqlalchemy.orm import Session

from yolo_agent.models.database import create_database_engine
from yolo_agent.models.dataset import utc_now
from yolo_agent.models.experiment import Experiment, ModelArtifact, ModelExport
from yolo_agent.services.datasets import DatasetError, owned_path
from yolo_agent.services.exports import fail_export
from yolo_agent.services.ultralytics_runtime import configure_ultralytics
from yolo_agent.settings import get_settings
from yolo_agent.utils.logging import configure_logging

logger = logging.getLogger(__name__)
SUFFIX = {"onnx": "onnx", "torchscript": "torchscript"}


def on_export_stopped(job, connection) -> None:
    """RQ 停止回调立即清理未交付的转换中间文件。"""
    on_export_failure(job, connection, RuntimeError, "操作者终止导出", None)


def on_export_failure(job, connection, exc_type, exc_value, traceback) -> None:
    """进程异常或超时也要把导出记录置为失败。

    :param job: RQ 任务。
    :param connection: RQ 连接。
    :param exc_type: 异常类型。
    :param exc_value: 异常内容。
    :param traceback: 异常堆栈。
    :returns: 无。
    """
    settings = get_settings()
    engine = create_database_engine(settings)
    try:
        fail_export(engine, job.id, f"{getattr(exc_type,'__name__',str(exc_type))}: {exc_value}")
        from yolo_agent.services.garbage import collect_export_garbage
        collect_export_garbage(engine, settings, job.id)
    finally:
        engine.dispose()


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _shapes(value) -> list:
    """TorchScript 输出可能是张量、元组或嵌套列表，只记录形状。"""
    if hasattr(value, "shape"):
        return [list(value.shape)]
    if isinstance(value, (list, tuple)):
        return [shape for item in value for shape in _shapes(item)]
    return []


def _verify(fmt: str, path: Path, imgsz: int) -> dict:
    """用部署端运行时加载并推理零输入，转换成功但无法加载的文件不能交付。"""
    import numpy as np

    if fmt == "onnx":
        import onnxruntime

        session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        tensor = session.get_inputs()[0]
        shape = [dim if isinstance(dim, int) else 1 for dim in tensor.shape]
        outputs = session.run(None, {tensor.name: np.zeros(shape, dtype=np.float32)})
        return {"runtime": f"onnxruntime {onnxruntime.__version__}", "input": {"name": tensor.name, "shape": shape}, "outputs": _shapes(outputs)}
    import torch

    module = torch.jit.load(str(path), map_location="cpu")
    with torch.no_grad():
        outputs = module(torch.zeros(1, 3, imgsz, imgsz))
    return {"runtime": f"torch {torch.__version__}", "input": {"shape": [1, 3, imgsz, imgsz]}, "outputs": _shapes(outputs)}


def run_export_task(export_id: str) -> dict:
    """在独立 Worker 进程中转换格式，原始注册权重始终只读。

    :param export_id: 已登记的导出任务。
    :returns: 导出文件指纹、大小与运行时校验结果。
    """
    settings = get_settings()
    configure_logging(settings, "export-task")
    job = get_current_job()
    if job is None or job.id != export_id:
        raise RuntimeError("导出只能由对应 RQ 任务执行")
    engine = create_database_engine(settings)
    try:
        with Session(engine) as session, session.begin():
            claimed = session.execute(update(ModelExport).where(ModelExport.id == export_id, ModelExport.status == "queued").values(status="running", started_at=utc_now()))
            if claimed.rowcount != 1:
                raise RuntimeError("导出任务已被执行或状态不允许启动")
            record = session.get(ModelExport, export_id)
            model = session.get(ModelArtifact, record.model_id)
            experiment = session.get(Experiment, model.experiment_id)
            fmt, kind, model_id, weight, digest = record.format, model.kind, model.id, model.path, model.sha256
            imgsz = int(experiment.parameters["imgsz"])
        source = owned_path(settings, weight)
        def progress(stage):
            from yolo_agent.services.task_control import require_running
            with Session(engine) as session, session.begin():
                current = session.get(ModelExport, export_id)
                require_running(current)
                current.verification = {"progress": stage}
        progress("verifying_weights")
        if not source.is_file() or _hash(source) != digest:
            raise DatasetError(409, "注册权重缺失或指纹变化")
        output = owned_path(settings, f"model_exports/{export_id}")
        output.mkdir(parents=True, exist_ok=True)
        # 注册目录中的权重只读，转换在副本上进行，避免训练库把产物写回注册目录。
        work = output / f"{kind}.pt"
        shutil.copy2(source, work)
        work.chmod(0o644)
        with (output / "export.log").open("a", buffering=1, encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
            progress("converting")
            configure_ultralytics(settings)
            from ultralytics import YOLO

            # simplify/dynamic 只属于 ONNX，训练库会拒绝其它格式携带这些参数。
            options = {"simplify": True, "dynamic": False} if fmt == "onnx" else {}
            produced = Path(YOLO(str(work)).export(format=fmt, imgsz=imgsz, device="cpu", **options))
        work.unlink(missing_ok=True)
        progress("verifying_runtime")
        verification = _verify(fmt, produced, imgsz)
        final = output / f"{kind}-{model_id[:8]}.{SUFFIX[fmt]}"
        produced.replace(final)
        final.chmod(0o444)
        result = {"path": final.relative_to(settings.storage_dir.resolve()).as_posix(), "sha256": _hash(final), "size_bytes": final.stat().st_size, "verification": verification}
        with Session(engine) as session, session.begin():
            record = session.get(ModelExport, export_id)
            from yolo_agent.services.task_control import require_running
            require_running(record)
            record.status, record.finished_at = "completed", utc_now()
            record.path, record.sha256, record.size_bytes, record.verification = result["path"], result["sha256"], result["size_bytes"], verification
        logger.info("模型格式导出完成", extra={"task_id": export_id, "result": result})
        return {"export_id": export_id, "format": fmt, **result}
    except Exception as error:
        fail_export(engine, export_id, f"{type(error).__name__}: {error}")
        from yolo_agent.services.garbage import collect_export_garbage
        collect_export_garbage(engine, settings, export_id)
        logger.exception("模型格式导出失败", extra={"task_id": export_id})
        raise
    finally:
        engine.dispose()
