"""原生部署的 CVAT 设置：官方生产配置把 OPA 地址写死为容器服务名，这里改指本机进程。"""

from cvat.settings.production import *  # noqa: F403

IAM_OPA_HOST = "http://127.0.0.1:8181"
IAM_OPA_DATA_URL = f"{IAM_OPA_HOST}/v1/data"
# 汉化版把副标题改在后端，原生部署沿用官方后端，在此补齐
ABOUT_INFO = {"subtitle": "开放数据标注平台"}
