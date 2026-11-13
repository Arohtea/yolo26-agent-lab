"""在隔离环境中安装经过明确选择的官方 PyTorch wheel。"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from packaging.version import InvalidVersion, Version

TORCH_VERSION = "2.10.0"
TORCHVISION_VERSION = "0.25.0"
# 使用 Toolkit 发布对应的保守驱动门槛，不依赖兼容模式放宽要求。
# 优先使用项目目标 CUDA 12.8，兼容 RTX 5060 Ti 的 Blackwell sm_120。
CUDA_DRIVERS = {"cu128": "570.26", "cu130": "580.65.06", "cu126": "560.28.03"}


def choose_flavor() -> tuple[str, str]:
    """安装前按宿主平台、驱动和 GPU 架构选择 wheel 来源。

    :returns: 安装类型与选择原因。
    """
    if platform.system() == "Darwin":
        return "macos", "macOS 使用官方原生 wheel，运行时默认 CPU"
    if platform.system() != "Linux":
        raise RuntimeError("第一阶段安装脚本仅支持 macOS 与 Linux")
    if not shutil.which("nvidia-smi"):
        return "cpu", "未发现 nvidia-smi，使用 CPU wheel"
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,compute_cap", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=15,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        return "cpu", f"NVIDIA 查询失败（{type(exc).__name__}），使用 CPU wheel；可核对驱动后显式选择 CUDA"
    rows = [line.split(",") for line in result.stdout.strip().splitlines() if line.strip()]
    if not rows:
        return "cpu", "没有可用 NVIDIA GPU，使用 CPU wheel"
    try:
        driver = min(Version(row[0].strip()) for row in rows)
        capability = min(Version(row[1].strip()) for row in rows)
        maximum_capability = max(Version(row[1].strip()) for row in rows)
    except (InvalidVersion, IndexError):
        return "cpu", "驱动或计算能力信息不完整，使用 CPU wheel；可核对硬件后显式选择 CUDA"
    for candidate in CUDA_DRIVERS:
        # 按整机架构范围筛选，混合显卡也不能选不覆盖其中某张卡的 wheel。
        if candidate == "cu126" and maximum_capability >= Version("10.0"):
            continue
        if candidate == "cu128" and capability < Version("7.0"):
            continue
        if candidate == "cu130" and capability < Version("7.5"):
            continue
        if driver >= Version(CUDA_DRIVERS[candidate]):
            return candidate, f"驱动 {driver}、计算能力范围 {capability}–{maximum_capability} 满足 {candidate} 保守安装门槛"
    return "cpu", f"驱动 {driver}、计算能力范围 {capability}–{maximum_capability} 无匹配的 CUDA wheel，使用 CPU wheel"


def main() -> None:
    """通过白名单索引安装，不接受任意下载地址或额外 pip 参数。

    :returns: 无；安装失败时以非零状态退出。
    """
    parser = argparse.ArgumentParser(description="安装项目 PyTorch 计算依赖")
    # pypi-cu128：PyPI 上 Linux 默认 wheel 即 CUDA 12.8 构建，可走 PyPI 镜像；用于官方 CUDA 索引访问不稳定的服务器
    parser.add_argument("--flavor", choices=["auto", "macos", "cpu", *CUDA_DRIVERS, "pypi-cu128"], default="auto")
    parser.add_argument("--dry-run", action="store_true", help="只输出选择结果，不安装")
    parser.add_argument("--allow-system-python", action="store_true", help="仅供独立 Docker 镜像构建")
    args = parser.parse_args()
    conda_prefix = os.getenv("CONDA_PREFIX")
    if not args.allow_system_python and (
        not conda_prefix or os.getenv("CONDA_DEFAULT_ENV") == "base"
        or Path(sys.prefix).resolve() != Path(conda_prefix).resolve()
    ):
        parser.error("请先进入项目独立 Conda 环境，不得在 base 中安装")
    if args.flavor == "auto":
        flavor, reason = choose_flavor()
    else:
        flavor, reason = args.flavor, "显式选择安装类型；镜像构建阶段不读取宿主 GPU"
    if flavor == "macos" and platform.system() != "Darwin":
        parser.error("macos wheel 仅适用于 macOS")
    if platform.system() == "Darwin" and flavor != "macos":
        parser.error("macOS 请使用 auto 或 macos，运行设备由 YOLO_DEVICE 控制")
    command = [sys.executable, "-m", "pip", "install", f"torch=={TORCH_VERSION}", f"torchvision=={TORCHVISION_VERSION}"]
    if flavor not in ("macos", "pypi-cu128"):
        command.extend(["--index-url", f"https://download.pytorch.org/whl/{flavor}"])
    print(json.dumps({"flavor": flavor, "reason": reason, "command": command}, ensure_ascii=False), flush=True)
    if not args.dry_run:
        subprocess.run(command, check=True)
        # PyPI 镜像回退同样核对真实构建，不能把选择名称当成已安装 CUDA 版本。
        expected_cuda = {"cu126": "12.6", "cu128": "12.8", "pypi-cu128": "12.8", "cu130": "13.0"}.get(flavor)
        verification = (
            "import json, torch, torchvision; "
            f"assert torch.__version__.split('+')[0] == {TORCH_VERSION!r}, 'PyTorch 版本不符'; "
            f"assert torchvision.__version__.split('+')[0] == {TORCHVISION_VERSION!r}, 'torchvision 版本不符'; "
            f"assert torch.version.cuda == {expected_cuda!r}, '安装的 CUDA runtime 与选择类型不符'; "
            "print(json.dumps({'torch': torch.__version__, 'torchvision': torchvision.__version__, 'cuda_runtime': torch.version.cuda}))"
        )
        subprocess.run([sys.executable, "-c", verification], check=True)


if __name__ == "__main__":
    main()
