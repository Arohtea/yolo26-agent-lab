"""白名单中的真实基础设施诊断。"""

import logging
import os

import mlflow
from rq import get_current_job

from yolo_agent.services.runtime import collect_runtime, require_device
from yolo_agent.settings import get_settings
from yolo_agent.utils.logging import configure_logging
from yolo_agent.workers.queue import RUNTIME_PREFIX, redis_connection


def infrastructure_diagnostic() -> dict:
    """在独立任务进程执行设备运算并保存可追踪诊断工件。

    :returns: 实际设备、执行进程、MLflow run 和诊断工件信息。
    :raises RuntimeError: 计算设备或外部服务不可用。
    """
    settings = get_settings()
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
    os.environ.setdefault("MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT", "30")
    configure_logging(settings, "task")
    logger = logging.getLogger(__name__)
    job = get_current_job()
    if job is None:
        raise RuntimeError("诊断必须通过 RQ Worker 执行")
    try:
        report = collect_runtime(settings.device)
        device = require_device(report)

        # 真正执行一次设备运算，避免把枚举成功当成 CUDA 可执行的证据。
        import torch

        try:
            value = torch.ones(8, device=device).sum().item()
        except RuntimeError as exc:
            if settings.device != "auto" or device == "cpu":
                raise
            report.warnings.append(f"CUDA 实际运算失败，自动回退 CPU：{exc}")
            report.effective_device = device = "cpu"
            report.selection_reason = "CUDA 运算验证失败，自动回退 CPU"
            value = torch.ones(8, device=device).sum().item()

        artifact = settings.storage_dir / "diagnostics" / f"{job.id}.json"
        artifact.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment("infrastructure-diagnostics")
        with mlflow.start_run(tags={"kind": "infrastructure_diagnostic", "is_training": "false", "rq_job_id": job.id}) as run:
            mlflow.log_params({"device": device, "platform": report.os, "torch_version": report.torch_version})
            mlflow.log_metric("tensor_sum", value)
            mlflow.log_artifact(str(artifact), artifact_path="diagnostics")
            result = {
                "kind": "infrastructure_diagnostic", "is_training": False,
                "job_id": job.id, "device": device, "tensor_sum": value,
                "process_id": os.getpid(), "worker_name": job.worker_name,
                "mlflow_run_id": run.info.run_id,
                "artifact_uri": f"runs:/{run.info.run_id}/diagnostics/{artifact.name}",
                "runtime": report.model_dump(mode="json"),
            }
        with redis_connection(settings) as connection:
            connection.setex(f"{RUNTIME_PREFIX}{job.worker_name}", settings.runtime_report_ttl, report.model_dump_json())
        logger.info("诊断完成", extra={"task_id": job.id, "device": device, "mlflow_run_id": result["mlflow_run_id"]})
        return result
    except Exception:
        logger.exception("诊断失败，交由 RQ 保存失败状态，不自动无限重试", extra={"task_id": job.id})
        raise
