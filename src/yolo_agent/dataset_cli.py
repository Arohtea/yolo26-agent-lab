"""通过正式 HTTP 契约操作数据集，命令行不绕过异步执行面。"""

import argparse
import json
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import httpx

from yolo_agent.settings import Settings


def register_commands(subparsers) -> None:
    """为现有管理入口增加数据集命令。

    :param subparsers: argparse 子命令注册器。
    :returns: 无。
    """
    upload = subparsers.add_parser("import-images", help="通过 API 上传目录中的图片并等待检查")
    upload.add_argument("--directory", required=True, type=Path)
    upload.add_argument("--metadata", type=Path, help="按相对图片路径提供采集分组的 JSON 文件")
    analyze = subparsers.add_parser("analyze-dataset", help="提交真实数据质量分析")
    version = subparsers.add_parser("create-version", help="创建不可变的分组划分版本")
    version.add_argument("--name", default="数据集快照")
    version.add_argument("--seed", type=int, default=42)
    version.add_argument("--train-ratio", type=float, default=0.8)
    version.add_argument("--val-ratio", type=float, default=0.1)
    version.add_argument("--test-ratio", type=float, default=0.1)
    version.add_argument("--group-by", nargs="*", choices=["video_id", "scene_id", "batch_id"], default=["video_id", "scene_id", "batch_id"])
    for parser in (analyze, version):
        parser.add_argument("--phash-threshold", type=int, default=4)
    for parser in (upload, analyze, version):
        parser.add_argument("--dataset-id", required=True)
        parser.add_argument("--api-url", help="API 根地址，默认使用当前配置的本机监听端口")
        parser.add_argument("--wait-seconds", type=int, default=1800)
        parser.set_defaults(handler=run_command)


def run_command(args: argparse.Namespace, settings: Settings) -> None:
    """调用 API 并有界等待任务，失败时保留服务器返回的原因。

    :param args: 已解析的命令参数。
    :param settings: 本机连接地址和上传限制。
    :returns: 无；成功时输出持久任务记录。
    """
    if args.wait_seconds <= 0:
        raise ValueError("等待时间必须大于零")
    host = "127.0.0.1" if settings.api_host == "0.0.0.0" else settings.api_host
    base = (args.api_url or f"http://{host}:{settings.api_port}").rstrip("/")
    with httpx.Client(base_url=base, timeout=120, trust_env=False) as client:
        endpoint = f"/api/v1/datasets/{args.dataset_id}"
        if args.command == "import-images":
            if not args.directory.is_dir():
                raise ValueError("导入目录不存在")
            paths = sorted(path for path in args.directory.rglob("*") if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
            if not paths or len(paths) > settings.max_upload_files:
                raise ValueError(f"目录中需包含 1 至 {settings.max_upload_files} 张支持格式图片，请分批导入")
            metadata = args.metadata.read_text(encoding="utf-8") if args.metadata else "{}"
            with ExitStack() as stack:
                files = [("files", (path.relative_to(args.directory).as_posix(), stack.enter_context(path.open("rb")))) for path in paths]
                response = client.post(f"{endpoint}/imports", files=files, data={"metadata": metadata})
        else:
            policy = {"phash_threshold": args.phash_threshold}
            suffix = "analyses"
            if args.command == "create-version":
                suffix = "versions"
                policy.update({key: getattr(args, key) for key in ("name", "seed", "train_ratio", "val_ratio", "test_ratio", "group_by")})
            response = client.post(f"{endpoint}/{suffix}", json=policy)
        if response.is_error:
            raise RuntimeError(f"API 返回 {response.status_code}：{response.text}")
        task_id = response.json()["id"]
        print(f"已提交数据集任务：{task_id}", file=sys.stderr)
        deadline = time.monotonic() + args.wait_seconds
        while time.monotonic() < deadline:
            current = client.get(f"/api/v1/dataset-tasks/{task_id}")
            current.raise_for_status()
            task = current.json()
            if task["status"] == "completed":
                print(json.dumps(task, ensure_ascii=False, indent=2))
                return
            if task["status"] == "failed":
                raise RuntimeError(f"任务 {task_id} 失败：{task['error']}")
            time.sleep(0.5)
        raise TimeoutError(f"等待任务 {task_id} 超时，任务未取消，请查询服务器任务记录")
