"""本地环境与容器共用的管理入口。"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from yolo_agent.dataset_cli import register_commands

from yolo_agent.services.runtime import collect_runtime, require_device
from yolo_agent.settings import get_settings
from yolo_agent.utils.logging import configure_logging


def main() -> None:
    """解析固定管理命令，禁止传入任意任务函数或 Shell。

    :returns: 无；失败时以非零退出码结束进程。
    """
    parser = argparse.ArgumentParser(description="YOLO26 Agent Lab 基础服务管理")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("api", help="启动 HTTP 控制面")
    subparsers.add_parser("worker", help="启动独立计算 Worker")
    subparsers.add_parser("agent-controller", help="推进持久化自主流程，不在节点中执行训练")
    subparsers.add_parser("init-db", help="升级数据库到当前迁移版本")
    runtime = subparsers.add_parser("runtime", help="输出真实环境与设备选择结果")
    runtime.add_argument("--device", help="auto、cpu、cuda 或 cuda:N")
    doctor = subparsers.add_parser("doctor", help="入队真实诊断任务并等待结果")
    doctor.add_argument("--wait-seconds", type=int, default=180, help="等待任务的时间上限")
    doctor.add_argument("--output", type=Path, help="诊断结果 JSON 保存路径")
    register_commands(subparsers)
    args = parser.parse_args()
    if getattr(args, "device", None):
        os.environ["YOLO_DEVICE"] = args.device
    try:
        settings = get_settings()
        if args.command == "api":
            import uvicorn
            from yolo_agent.api.main import app

            configure_logging(settings, "api")
            uvicorn.run(app, host=settings.api_host, port=settings.api_port, log_config=None)
        elif args.command == "worker":
            from yolo_agent.workers.runner import run_worker

            run_worker(settings)
        elif args.command == "agent-controller":
            from yolo_agent.agents.controller import run_controller
            run_controller(settings)
        elif args.command == "init-db":
            from alembic import command
            from alembic.config import Config

            configure_logging(settings, "migration")
            command.upgrade(Config("alembic.ini"), "head")
            print(json.dumps({"status": "upgraded", "database_url": settings.database_url}, ensure_ascii=False))
        elif args.command == "runtime":
            report = collect_runtime(settings.device)
            print(report.model_dump_json(indent=2))
            require_device(report)
        elif args.command == "doctor":
            from yolo_agent.workers.queue import enqueue_diagnostic

            if args.wait_seconds <= 0:
                raise ValueError("等待时长必须大于零")
            job = enqueue_diagnostic(settings)
            print(f"已提交真实诊断任务：{job.id}", file=sys.stderr)
            deadline = time.monotonic() + args.wait_seconds
            while time.monotonic() < deadline:
                status = job.get_status(refresh=True)
                if status == "finished":
                    result = job.return_value()
                    payload = json.dumps(result, ensure_ascii=False, indent=2)
                    if args.output:
                        args.output.parent.mkdir(parents=True, exist_ok=True)
                        args.output.write_text(payload + "\n", encoding="utf-8")
                    print(payload)
                    return
                if status in {"failed", "stopped", "canceled"}:
                    raise RuntimeError(f"诊断任务 {job.id} 状态为 {status}，请查看 Worker 日志与 RQ 失败记录")
                time.sleep(0.5)
            raise TimeoutError(f"等待任务 {job.id} 超时；任务未取消，可通过 RQ 查询状态")
        elif hasattr(args, "handler"):
            args.handler(args, settings)
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
