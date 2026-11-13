#!/bin/sh
# 原生 CVAT 进程入口：加载统一环境后执行命令。
# --wait 复刻官方 backend_entrypoint.sh run 的门禁：依赖端口可连且数据库、Redis 迁移完成后才启动。
here=$(cd "$(dirname "$0")" && pwd)
# 调用方可能处于项目 conda 环境；shapely 1.x 见到 CONDA_PREFIX 会去 venv 下寻找 GEOS 而导入失败
unset CONDA_PREFIX
set -a
. "$here/cvat.env"
set +a
export PYTHONPATH="$here"
cd "$HOME" || exit 1
if [ "$1" = "--wait" ]; then
    shift
    ./wait_for_deps.sh true
    until python3 manage.py migrate --check >/dev/null 2>&1 && python3 manage.py migrateredis --check >/dev/null 2>&1; do
        echo "等待 CVAT 迁移完成..."
        sleep 10
    done
fi
exec "$@"
