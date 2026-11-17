"""按执行进程实际能力选择 CPU 或 NVIDIA CUDA。"""

import logging
import os
import platform
import sys
from datetime import datetime, timezone
from typing import Any

import psutil
import pynvml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class GPUInfo(BaseModel):
    """仅列出 PyTorch 可见的逻辑设备，避免误用物理编号。"""

    logical_index: int
    name: str
    uuid: str | None = None
    memory_total_bytes: int
    memory_used_bytes: int | None = None
    memory_metric_source: str | None = None
    unavailable_reason: str | None = None


class RuntimeReport(BaseModel):
    """环境快照；空值表示指标不可获取，零值表示真实的零。"""

    collected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    hostname: str = Field(default_factory=platform.node)
    process_id: int = Field(default_factory=os.getpid)
    os: str = Field(default_factory=platform.system)
    os_version: str = Field(default_factory=platform.release)
    architecture: str = Field(default_factory=platform.machine)
    cpu: str = Field(default_factory=lambda: platform.processor() or platform.machine())
    cpu_count: int | None = Field(default_factory=lambda: os.cpu_count())
    memory_total_bytes: int = Field(default_factory=lambda: psutil.virtual_memory().total)
    python_version: str = Field(default_factory=platform.python_version)
    python_executable: str = sys.executable
    conda_environment: str | None = Field(default_factory=lambda: os.getenv("CONDA_DEFAULT_ENV"))
    torch_version: str | None = None
    cuda_runtime_version: str | None = None
    driver_version: str | None = None
    driver_cuda_max_version: str | None = None
    cuda_visible_devices: str | None = Field(default_factory=lambda: os.getenv("CUDA_VISIBLE_DEVICES"))
    cuda_available: bool = False
    gpu_count: int | None = None
    physical_gpu_count: int | None = None
    requested_device: str
    effective_device: str | None = None
    selection_reason: str = ""
    selection_error: str | None = None
    gpus: list[GPUInfo] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    queue_name: str | None = None
    worker_role: str | None = None


class DeviceUnavailableError(RuntimeError):
    """显式设备请求不可满足，调用者不得继续提交计算。"""


def collect_runtime(requested_device: str) -> RuntimeReport:
    """采集当前进程环境并执行统一设备策略，不分配计算张量。

    :param requested_device: 已通过配置校验的 auto、cpu、cuda 或 cuda:N。
    :returns: 包含选择结果、缺失指标原因及物理设备诊断的快照。
    """
    report = RuntimeReport(requested_device=requested_device)
    nvml: Any = None
    try:
        pynvml.nvmlInit()
        nvml = pynvml
        report.physical_gpu_count = nvml.nvmlDeviceGetCount()
        report.driver_version = nvml.nvmlSystemGetDriverVersion()
        try:
            version = nvml.nvmlSystemGetCudaDriverVersion_v2()
            report.driver_cuda_max_version = f"{version // 1000}.{version % 1000 // 10}"
        except pynvml.NVMLError as exc:
            report.warnings.append(f"无法获取驱动支持的 CUDA 上限：{exc}")
    except (pynvml.NVMLError, OSError, RuntimeError) as exc:
        report.warnings.append(f"NVML 指标不可用：{exc}")

    try:
        import torch

        report.torch_version = str(torch.__version__)
        report.cuda_runtime_version = torch.version.cuda
        try:
            report.cuda_available = bool(torch.version.cuda and torch.cuda.is_available())
            report.gpu_count = torch.cuda.device_count() if report.cuda_available else 0

            # 逻辑设备枚举完全由 PyTorch 决定，NVML 仅按 UUID 补充显存指标。
            for index in range(report.gpu_count):
                props = torch.cuda.get_device_properties(index)
                uuid = str(props.uuid) if getattr(props, "uuid", None) else None
                gpu = GPUInfo(
                    logical_index=index, name=props.name,
                    uuid=uuid, memory_total_bytes=props.total_memory,
                )
                if nvml is not None and uuid:
                    try:
                        handle = nvml.nvmlDeviceGetHandleByUUID(uuid)
                        gpu.memory_used_bytes = nvml.nvmlDeviceGetMemoryInfo(handle).used
                        gpu.memory_metric_source = "nvml_device_used"
                    except (nvml.NVMLError, ValueError) as exc:
                        gpu.unavailable_reason = f"该 UUID 的 NVML 显存不可用：{exc}"
                else:
                    gpu.unavailable_reason = "NVML 或设备 UUID 不可用，不能准确关联物理显存"
                report.gpus.append(gpu)
        except (RuntimeError, AssertionError, OSError) as exc:
            report.cuda_available = False
            report.gpu_count = None
            report.gpus = []
            report.warnings.append(f"CUDA 枚举失败，设备数量未知：{exc}")

        if requested_device == "cpu" or (requested_device == "auto" and report.os == "Darwin"):
            report.effective_device = "cpu"
            report.selection_reason = "显式选择 CPU" if requested_device == "cpu" else "macOS 自动模式按项目策略使用 CPU"
        elif requested_device == "auto":
            report.effective_device = "cuda:0" if report.cuda_available and report.gpu_count else "cpu"
            report.selection_reason = "NVIDIA CUDA 可用，选择逻辑 GPU 0" if report.effective_device != "cpu" else "未发现可用 NVIDIA CUDA，自动回退 CPU"
            if report.effective_device == "cpu":
                report.warnings.append(report.selection_reason)
        else:
            index = int(requested_device.split(":")[1]) if ":" in requested_device else 0
            if not report.cuda_available or index >= (report.gpu_count or 0):
                count = report.gpu_count if report.gpu_count is not None else "未知数量的"
                report.selection_error = f"请求 {requested_device}，但当前进程仅有 {count} 个可用 CUDA 设备"
                report.selection_reason = "显式 CUDA 请求不可满足，禁止降级执行"
            else:
                report.effective_device = f"cuda:{index}"
                report.selection_reason = "使用显式指定的 CUDA 逻辑设备"
    except (ImportError, OSError, RuntimeError) as exc:
        report.selection_error = f"PyTorch 环境检测失败：{exc}"
        report.selection_reason = "无法验证计算环境"
    finally:
        if nvml is not None:
            nvml.nvmlShutdown()

    if report.physical_gpu_count and not report.cuda_available:
        report.warnings.append("检测到物理 NVIDIA GPU，但 PyTorch CUDA 不可用；请检查 wheel、驱动及容器设备授权")
    return report


def require_device(report: RuntimeReport) -> str:
    """在任务执行边界拒绝不可用设备。

    :param report: 当前执行进程重新采集的环境快照。
    :returns: 可传给 PyTorch 的实际设备名称。
    :raises DeviceUnavailableError: 显式设备请求或运行环境不满足要求。
    """
    if report.selection_error or report.effective_device is None:
        raise DeviceUnavailableError(report.selection_error or "设备选择失败")
    return report.effective_device
