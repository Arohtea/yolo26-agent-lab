"""操作者将已上传、已审核的 YOLO 数据包送入固定 RQ 导入任务。"""

import argparse
import json
import os
from rq import Callback

from sqlalchemy import select
from sqlalchemy.orm import Session

from yolo_agent.models.database import create_database_engine
from yolo_agent.models.dataset import Dataset, DatasetTask, new_id
from yolo_agent.services.datasets import owned_path
from yolo_agent.settings import get_settings
from yolo_agent.workers.queue import get_queue


def main():
    if not os.getenv("CONDA_PREFIX") or os.getenv("CONDA_DEFAULT_ENV") == "base":
        raise RuntimeError("必须使用项目 Conda 环境")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--source", required=True, help="storage 内已上传数据包目录")
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--confirmation", required=True)
    parser.add_argument("--accept-reviewed", action="store_true", required=True)
    args = parser.parse_args()
    settings = get_settings()
    source = owned_path(settings, args.source)
    if not (source / "source_manifest.json").is_file():
        raise ValueError("缺少数据包来源与逐文件指纹清单")
    engine = create_database_engine(settings)
    with Session(engine) as session, session.begin():
        if session.get(Dataset, args.dataset_id) is None:
            raise ValueError("目标数据集不存在")
        if session.scalar(select(DatasetTask.id).where(DatasetTask.dataset_id == args.dataset_id)):
            raise ValueError("目标数据集已有任务，请检查现有状态；不能重复导入")
        identity = new_id()
        session.add(DatasetTask(id=identity, dataset_id=args.dataset_id, kind="reviewed_import", total=0,
            parameters={"source":args.source,"reviewer":args.reviewer,"confirmation":args.confirmation,"accepted_reviewed_source":True}))
    try:
        get_queue(settings).enqueue("yolo_agent.workers.reviewed_import.run_reviewed_import", identity,
            job_id=identity, job_timeout=settings.dataset_job_timeout,
            result_ttl=settings.job_result_ttl, failure_ttl=settings.job_result_ttl,
            on_failure=Callback("yolo_agent.workers.dataset_tasks.on_dataset_failure"),
            description="导入操作者确认的已审核 YOLO 数据（保留划分）")
    except Exception as error:
        from yolo_agent.services.datasets import fail_dataset_task
        fail_dataset_task(engine, identity, f"导入入队失败：{type(error).__name__}")
        raise
    print(json.dumps({"task_id":identity,"dataset_id":args.dataset_id},ensure_ascii=False))
    engine.dispose()


if __name__ == "__main__":
    main()
