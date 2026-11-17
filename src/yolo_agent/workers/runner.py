"""使用成熟的 SpawnWorker 避免 macOS/CUDA 的 fork 限制。"""

import logging
import os
import platform
import threading
import uuid

from rq import SpawnWorker
from rq.serializers import JSONSerializer

from yolo_agent.services.runtime import collect_runtime, require_device
from yolo_agent.settings import Settings
from yolo_agent.utils.logging import configure_logging
from yolo_agent.workers.queue import RUNTIME_PREFIX, get_queue, redis_connection


def run_worker(settings: Settings, *, queue_name: str | None = None) -> None:
    """注册环境快照并持续消费计算队列。

    :param settings: 当前 Worker 自身的环境配置。
    :returns: 收到停止信号并结束消费后返回。
    """
    configure_logging(settings, "worker")
    logger = logging.getLogger(__name__)
    queue_name = queue_name or settings.queue_name
    role = "annotation" if queue_name == settings.annotation_queue_name else "data" if queue_name == settings.data_queue_name else "compute"
    report = collect_runtime(settings.device)
    device = require_device(report)
    name = f"{platform.node()}-{uuid.uuid4().hex[:8]}"
    stop = threading.Event()
    connection = redis_connection(settings, blocking=True)
    report_connection = redis_connection(settings)

    def publish_reports() -> None:
        """报告设置 TTL，Worker 退出后过期记录不会被误认作在线。"""
        while not stop.is_set():
            try:
                current = collect_runtime(settings.device)
                current.queue_name, current.worker_role = queue_name, role
                report_connection.setex(f"{RUNTIME_PREFIX}{name}", settings.runtime_report_ttl, current.model_dump_json())
            except Exception:
                logger.exception("Worker 环境报告失败，等待下一次心跳")
            stop.wait(settings.runtime_report_interval)

    thread = threading.Thread(target=publish_reports, name="runtime-reporter", daemon=True)
    thread.start()
    logger.info("Worker 启动", extra={"device": device, "worker_name": name, "queue": queue_name, "role": role, "warnings": report.warnings})
    try:
        worker = SpawnWorker(
            [get_queue(settings, queue_name=queue_name)], connection=connection, name=name,
            serializer=JSONSerializer,
        )
        worker.work()
    finally:
        stop.set()
        thread.join(timeout=settings.health_timeout + 1)
        try:
            report_connection.delete(f"{RUNTIME_PREFIX}{name}")
        except Exception:
            logger.warning("无法删除退出报告，将由 Redis TTL 自动清理", exc_info=True)
        finally:
            connection.close()
            report_connection.close()


def run_annotation_worker(settings: Settings) -> None:
    """复用 RQ SpawnWorker，设备与 CPU 线程配置显式传给新解释器。"""
    os.environ["YOLO_DEVICE"] = settings.annotation_device
    os.environ["OMP_NUM_THREADS"] = str(settings.annotation_cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(settings.annotation_cpu_threads)
    run_worker(settings.model_copy(update={"device": settings.annotation_device}), queue_name=settings.annotation_queue_name)
