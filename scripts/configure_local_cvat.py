"""配置本地 CVAT 账户，凭据只保存到受保护的环境文件。"""

import argparse
import json
import os
import secrets
import subprocess
from pathlib import Path
from urllib.parse import urlparse


def main() -> None:
    """在已启动的本地 CVAT 中创建账户，不覆盖已有账户密码。

    :returns: 无；只输出配置成功状态，不输出密码。
    """
    parser = argparse.ArgumentParser(description="配置本地 CVAT 操作账户")
    parser.add_argument("--native", action="store_true", help="服务器原生部署（configs/server），不经 Docker 容器")
    args = parser.parse_args()
    if not os.getenv("CONDA_PREFIX") or os.getenv("CONDA_DEFAULT_ENV") == "base":
        raise RuntimeError("请进入项目独立 Conda 环境")
    from yolo_agent.settings import get_settings

    settings = get_settings()
    if urlparse(settings.cvat_url).hostname not in {"localhost", "127.0.0.1"}:
        raise RuntimeError("该脚本仅用于本机 CVAT，不允许修改远程账户")
    username = settings.cvat_username or "yolo-operator"
    password = settings.cvat_password.get_secret_value() or secrets.token_urlsafe(24)
    payload = repr(json.dumps({"username":username,"password":password}))
    program = f"""
import json
from django.contrib.auth import get_user_model
config = json.loads({payload})
User = get_user_model()
if not User.objects.filter(username=config['username']).exists():
    User.objects.create_superuser(config['username'], 'operator@example.invalid', config['password'])
elif not User.objects.get(username=config['username']).check_password(config['password']):
    raise RuntimeError('账户已存在且密码不匹配，请配置现有凭据，不覆盖密码')
print('本地 CVAT 账户已就绪')
"""
    # 原生部署以 CVAT 进程同一用户和环境执行 manage.py，与 supervisord 中的服务一致
    runner = os.path.abspath("configs/server/cvat-run.sh")
    command = ["su","django","-s","/bin/sh","-c",f"{runner} python3 manage.py shell"] if args.native else ["docker","exec","-i","cvat_server","python","/home/django/manage.py","shell"]
    result = subprocess.run(command, input=program, text=True, capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError(f"CVAT 账户配置失败，请检查 CVAT 是否已完成数据库初始化：{result.stderr.strip()[-800:]}")
    path = Path(".env")
    entries = {"YOLO_CVAT_URL":settings.cvat_url,"YOLO_CVAT_USERNAME":username,"YOLO_CVAT_PASSWORD":password}
    lines = [line for line in path.read_text().splitlines() if not any(line.startswith(key+"=") for key in entries)] if path.exists() else []
    path.write_text("\n".join(lines+[f"{key}={value}" for key,value in entries.items()])+"\n")
    path.chmod(0o600)
    print("本地 CVAT 账户已配置，凭据保存在 .env；请重启 API 和 Worker 使配置生效")


if __name__ == "__main__":
    main()
