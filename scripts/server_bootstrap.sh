#!/usr/bin/env bash
# 无 Docker 的 GPU 服务器（K8s 容器）部署与恢复，可重复执行。
# 根文件系统上的 apt 依赖、django 用户和路径链接在容器重建后会丢失，数据都在 /data 持久卷；
# 首次部署前先按 docs/deployment.md 在本机打包 CVAT 构件并上传到 /data/cvat-artifacts。
set -euo pipefail
cd "$(dirname "$0")/.."
# files.pythonhosted.org 在该服务器上几乎不可用，默认走清华 PyPI 镜像
export PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}
CONDA=/data/miniconda
PG=$CONDA/envs/cvat-services/bin
APP=$CONDA/envs/yolo26-agent-lab/bin

# 官方 cvat/server:v2.64.0 运行阶段的 apt 依赖；镜像与宿主同为 Ubuntu 22.04 / Python 3.10，可直接复用其 venv
if ! dpkg -s nginx wait-for-it libxmlsec1-openssl zstd >/dev/null 2>&1; then
    apt-get update -q
    DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
        bzip2 ca-certificates curl git libgeos-c1v5 libgl1 libgomp1 libldap-2.5-0 libpython3.10 libsasl2-2 \
        libxml2 libxmlsec1 libxmlsec1-openssl nginx p7zip-full poppler-utils python3 python3-venv tzdata \
        unrar wait-for-it zstd rsync
fi
id django >/dev/null 2>&1 || useradd -u 1000 -U -M -d /home/django -s /bin/bash django

# CVAT 构件：venv 内的脚本以 /opt/venv 为绝对路径，代码与数据沿用镜像的 /home/django
if [ ! -d /data/cvat/venv ]; then
    mkdir -p /data/cvat/lib
    zstd -dc /data/cvat-artifacts/venv.tar.zst | tar -x -C /data/cvat
    tar -xf /data/cvat-artifacts/django.tar -C /data/cvat
    cp -a /data/cvat-artifacts/ffmpeg-lib/. /data/cvat/lib/
    cp -a /data/cvat-artifacts/bin /data/cvat-artifacts/ui /data/cvat/
    chown -R root:root /data/cvat/ui
    chown -R django:django /data/cvat/django /data/cvat/bin /data/cvat/lib
fi
ln -sfn /data/cvat/django /home/django
ln -sfn /data/cvat/venv /opt/venv
install -d -o django -g django /data/cvat/run /data/cvat/run/nginx /data/cvat/redis-inmem /data/cvat/kvrocks /data/cvat/django/logs/supervisor

# 依赖服务与官方 compose 版本一致：PostgreSQL 15、Redis 7.2.11（CVAT）与 7.4（项目队列）
[ -d $CONDA/envs/cvat-services ] || $CONDA/bin/conda create -y -q -p $CONDA/envs/cvat-services -c conda-forge --override-channels postgresql=15 redis-server=7.2.11
if [ ! -d $CONDA/envs/yolo26-services ]; then
    $CONDA/bin/conda create -y -q -p $CONDA/envs/yolo26-services -c conda-forge --override-channels python=3.11 redis-server=7.4.7 pip
    $CONDA/envs/yolo26-services/bin/pip install -q mlflow==3.18.0
fi
if [ ! -s /data/cvat/postgres/PG_VERSION ]; then
    install -d -o django -g django -m 700 /data/cvat/postgres
    # 与 compose 的 cvat_db 相同：root 用户、trust 认证、cvat 库；只经本机回环地址访问
    su django -s /bin/sh -c "cd /data/cvat && $PG/initdb -D /data/cvat/postgres -U root -A trust -E UTF8 --locale=C.UTF-8 \
        && $PG/pg_ctl -D /data/cvat/postgres -o '-c listen_addresses= -c unix_socket_directories=/data/cvat/run' -w start \
        && $PG/createdb -h /data/cvat/run -U root cvat && $PG/pg_ctl -D /data/cvat/postgres -w stop"
fi

if [ ! -x $APP/yolo-agent ]; then
    [ -d $CONDA/envs/yolo26-agent-lab ] || $CONDA/bin/conda create -y -q -n yolo26-agent-lab python=3.12 pip
    $APP/python -m pip install -q -e .
    # 目标计算栈固定 CUDA 12.8，覆盖 5060 Ti；优先使用官方索引并核对实际安装构建。
    $CONDA/bin/conda run --no-capture-output -p $CONDA/envs/yolo26-agent-lab python scripts/install_torch.py --flavor cu128
    # fiftyone-db 在 Linux 只有源码包，构建时从 MongoDB 官网下载数据库；服务器访问不了，改用本机上传并核对 SHA256 的官方压缩包
    if ! ls /data/wheels/fiftyone_db-1.4.1-*.whl >/dev/null 2>&1; then
        tmp=$(mktemp -d)
        $APP/python -m pip download -q --no-deps --no-binary :all: fiftyone-db==1.4.1 -d "$tmp"
        tar -xzf "$tmp/fiftyone_db-1.4.1.tar.gz" -C "$tmp"
        cp /data/cvat-artifacts/mongodb-linux-x86_64-ubuntu2204-7.0.28.tgz "$tmp/fiftyone_db-1.4.1/"
        $APP/python -m pip wheel -q --no-deps "$tmp/fiftyone_db-1.4.1" -w /data/wheels
    fi
    $APP/python -m pip install -q /data/wheels/fiftyone_db-1.4.1-*.whl
    $APP/python -m pip install -q -e '.[training,evaluation,agent,export]'
fi
mkdir -p storage/logs storage/redis storage/mlflow

# 已在运行的 supervisord 不重复启动
$APP/supervisorctl -c configs/server/cvat-supervisord.conf pid >/dev/null 2>&1 || $APP/supervisord -c configs/server/cvat-supervisord.conf
set +u  # conda 激活脚本会引用未定义变量
source $CONDA/bin/activate yolo26-agent-lab
set -u
yolo-agent init-db
supervisorctl -c configs/server/supervisord.conf pid >/dev/null 2>&1 || supervisord -c configs/server/supervisord.conf
echo "部署完成：supervisorctl -c configs/server/supervisord.conf status 查看项目进程，cvat-supervisord.conf 查看 CVAT"
