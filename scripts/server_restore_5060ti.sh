#!/usr/bin/env bash
# 恢复已部署的 Ubuntu 24.04 / RTX 5060 Ti 主机，不重新下载权重或重建环境。
set -euo pipefail
cd "$(dirname "$0")/.."
CONDA=/data/miniconda
APP=$CONDA/envs/torch/bin
MONGOD=/data/mongodb-linux-x86_64-ubuntu2204-7.0.28/bin/mongod

if [ "$(id -u)" != 0 ]; then
    echo '恢复系统依赖和服务用户需要 root。' >&2
    exit 1
fi
for path in "$APP/yolo-agent" /data/cvat/venv/bin/python3 /data/cvat/postgres/PG_VERSION "$MONGOD"; do
    if [ ! -e "$path" ]; then
        echo "缺少已部署资源：$path；本脚本用于恢复，不负责首次构建。" >&2
        exit 1
    fi
done

if ! dpkg -s nginx wait-for-it libgeos-c1t64 libxmlsec1t64-openssl >/dev/null 2>&1; then
    apt-get update -q
    DEBIAN_FRONTEND=noninteractive apt-get install -yq nginx wait-for-it libgeos-c1t64 libxmlsec1t64-openssl libgl1 libgomp1 tmux zstd
fi

# 持久化 PostgreSQL 数据的实际 UID 是服务用户身份的依据，避免容器重建后权限漂移。
django_uid=$(stat -c %u /data/cvat/postgres)
if ! id django >/dev/null 2>&1; then
    useradd -u "$django_uid" -U -M -d /home/django -s /bin/bash django
fi
if [ "$(id -u django)" != "$django_uid" ]; then
    echo 'django 用户与数据库目录 UID 不一致，请先处理用户身份；不自动更改数据库权限。' >&2
    exit 1
fi
ln -sfn /data/cvat/django /home/django
ln -sfn /data/cvat/venv /opt/venv
install -d -o django -g django /data/cvat/run /data/cvat/run/nginx /data/cvat/django/logs/supervisor
chown -R django:django /data/cvat/run/nginx /data/cvat/django/logs
if [ -f /data/cvat/run/nginx.pid ]; then
    chown django:django /data/cvat/run/nginx.pid
fi
su django -s /bin/sh -c "/usr/sbin/nginx -t -c $(pwd)/configs/server/cvat-nginx.conf"

set +u
source "$CONDA/bin/activate" torch
set -u
# Ubuntu 24.04 上 fiftyone-db 的安装结果可能不含数据库二进制，复用已校验的官方 MongoDB。
fiftyone_bin=$(python -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/fiftyone/db/bin")')
ln -sfn "$MONGOD" "$fiftyone_bin/mongod"
"$MONGOD" --version >/dev/null
mkdir -p storage/logs storage/redis storage/mlflow
yolo-agent init-db
if ! supervisorctl -c configs/server/cvat-supervisord.conf pid >/dev/null 2>&1; then
    supervisord -c configs/server/cvat-supervisord.conf
fi
if ! supervisorctl -c configs/server/supervisord.conf pid >/dev/null 2>&1; then
    supervisord -c configs/server/supervisord.conf
fi
supervisorctl -c configs/server/supervisord.conf status
supervisorctl -c configs/server/cvat-supervisord.conf status
