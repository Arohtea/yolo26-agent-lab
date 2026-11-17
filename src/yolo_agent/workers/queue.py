"""仅暴露已注册任务，不允许调用者指定任意 Python 函数。"""

from redis import Redis
from rq import Callback, Queue
from rq.job import Job
from rq.serializers import JSONSerializer

from yolo_agent.settings import Settings

RUNTIME_PREFIX = "yolo:runtime:"


def redis_connection(settings: Settings, *, blocking: bool = False) -> Redis:
    """建立有界超时的 Redis 连接。

    :param settings: Redis 地址与健康检查超时。
    :param blocking: 仅 Worker 阻塞取队列时关闭读超时，RQ 自身管理等待窗口。
    :returns: 不自动解码的连接，兼容 RQ 二进制数据。
    """
    return Redis.from_url(
        settings.redis_url, socket_connect_timeout=settings.health_timeout,
        socket_timeout=None if blocking else settings.health_timeout, health_check_interval=30,
    )


def get_queue(settings: Settings, *, queue_name: str | None = None) -> Queue:
    """让生产者和消费者采用同一 JSON 编码规则。

    :param settings: 队列及执行超时配置。
    :returns: 项目计算任务队列。
    """
    return Queue(
        queue_name or settings.queue_name, connection=redis_connection(settings),
        serializer=JSONSerializer, default_timeout=settings.job_timeout,
    )


def enqueue_diagnostic(settings: Settings) -> Job:
    """提交固定的基础设施诊断任务。

    :param settings: 队列连接及保留策略。
    :returns: 可查询状态的 RQ 任务。
    """
    return get_queue(settings).enqueue(
        "yolo_agent.workers.tasks.infrastructure_diagnostic", job_timeout=settings.job_timeout,
        result_ttl=settings.job_result_ttl, failure_ttl=settings.job_result_ttl,
        description="基础设施真实诊断（非训练实验）",
    )


def enqueue_dataset_task(settings: Settings, task_id: str) -> Job:
    """仅允许已在业务库登记的固定数据集任务入口。

    :param settings: 队列连接、执行时限及结果保留策略。
    :param task_id: Worker 从数据库读取的任务 ID，不接受函数名或文件路径。
    :returns: 已提交的 RQ 任务。
    """
    return get_queue(settings, queue_name=settings.data_queue_name).enqueue(
        "yolo_agent.workers.dataset_tasks.run_dataset_task", task_id,
        job_id=task_id, job_timeout=settings.dataset_job_timeout,
        result_ttl=settings.job_result_ttl, failure_ttl=settings.job_result_ttl,
        on_failure=Callback("yolo_agent.workers.dataset_tasks.on_dataset_failure"),
        on_stopped=Callback("yolo_agent.workers.dataset_tasks.on_dataset_stopped"),
        description=f"数据集处理 {task_id}",
    )


def enqueue_annotation_task(settings: Settings, task_id: str, *, cpu_only: bool = False) -> Job:
    """提交固定标注执行器，禁用调用者选择函数或下载地址。

    :param settings: 执行时限及队列配置。
    :param task_id: 已登记的业务任务 ID。
    :returns: RQ 任务。
    """
    return get_queue(settings, queue_name=settings.data_queue_name if cpu_only else settings.annotation_queue_name).enqueue(
        "yolo_agent.workers.annotation_tasks.run_annotation_task", task_id,
        job_id=task_id, job_timeout=settings.annotation_job_timeout,
        result_ttl=settings.job_result_ttl, failure_ttl=settings.job_result_ttl,
        on_failure=Callback("yolo_agent.workers.annotation_tasks.on_annotation_failure"),
        on_stopped=Callback("yolo_agent.workers.dataset_tasks.on_dataset_stopped"),
        description=f"标注处理 {task_id}",
    )


def enqueue_training_task(settings: Settings, task_id: str, *, timeout_seconds: int | None = None) -> Job:
    """提交固定训练入口，任务不接受外部命令、脚本或模型地址。

    :param settings: 队列和最长训练时限。
    :param task_id: 已登记的训练任务。
    :param timeout_seconds: 编排预算收紧后的硬超时。
    :returns: RQ 任务。
    """
    return get_queue(settings).enqueue(
        "yolo_agent.workers.training_tasks.run_training_task", task_id,
        job_id=task_id, job_timeout=min(settings.training_job_timeout,timeout_seconds or settings.training_job_timeout),
        result_ttl=settings.job_result_ttl, failure_ttl=settings.job_result_ttl,
        on_failure=Callback("yolo_agent.workers.training_tasks.on_training_failure"),
        description=f"YOLO26 训练 {task_id}",
    )


def enqueue_evaluation_task(settings: Settings, task_id: str, *, timeout_seconds: int | None = None) -> Job:
    """只提交固定的验证集评估入口。

    :param settings: 队列与任务时限。
    :param task_id: 已登记评估任务。
    :param timeout_seconds: 内部工作流收紧的时间限制。
    :returns: RQ 任务。
    """
    return get_queue(settings).enqueue(
        "yolo_agent.workers.evaluation_tasks.run_evaluation_task", task_id,
        job_id=task_id, job_timeout=min(settings.evaluation_job_timeout,timeout_seconds or settings.evaluation_job_timeout),
        result_ttl=settings.job_result_ttl, failure_ttl=settings.job_result_ttl,
        on_failure=Callback("yolo_agent.workers.evaluation_tasks.on_evaluation_failure"),
        on_stopped=Callback("yolo_agent.workers.evaluation_tasks.on_evaluation_stopped"),
        description=f"验证集评估 {task_id}",
    )


def enqueue_export_task(settings: Settings, export_id: str) -> Job:
    """只提交固定的部署格式转换入口，格式与权重都从业务记录读取。

    :param settings: 队列与转换时限。
    :param export_id: 已登记的导出任务。
    :returns: RQ 任务。
    """
    return get_queue(settings, queue_name=settings.data_queue_name).enqueue(
        "yolo_agent.workers.export_tasks.run_export_task", export_id,
        job_id=export_id, job_timeout=settings.export_job_timeout,
        result_ttl=settings.job_result_ttl, failure_ttl=settings.job_result_ttl,
        on_failure=Callback("yolo_agent.workers.export_tasks.on_export_failure"),
        on_stopped=Callback("yolo_agent.workers.export_tasks.on_export_stopped"),
        description=f"模型格式导出 {export_id}",
    )
