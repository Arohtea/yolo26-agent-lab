"""导出当前实际环境，不把另一平台的锁定文件当作安装结果。"""

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch

from yolo_agent.services.runtime import collect_runtime


def main() -> None:
    """记录已安装依赖、解释器和计算 wheel 类型。

    :returns: 无；文件写入 docs/environments 并输出路径。
    """
    if not os.getenv("CONDA_PREFIX") or os.getenv("CONDA_DEFAULT_ENV") == "base":
        raise RuntimeError("必须在项目独立 Conda 环境中导出依赖")
    flavor = f"cu{torch.version.cuda.replace('.', '')}" if torch.version.cuda else "cpu"
    name = f"{platform.system().lower()}-{platform.machine()}-{flavor}"
    directory = Path("docs/environments")
    directory.mkdir(parents=True, exist_ok=True)
    frozen = subprocess.run(
        [sys.executable, "-m", "pip", "list", "--format=freeze", "--exclude-editable"],
        check=True, capture_output=True, text=True,
    ).stdout
    (directory / f"{name}.requirements.txt").write_text(frozen, encoding="utf-8")
    report = collect_runtime("auto")
    (directory / f"{name}.json").write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    explicit = subprocess.run(
        ["conda", "list", "--explicit"], check=True, capture_output=True, text=True,
    ).stdout
    (directory / f"{name}.conda.txt").write_text(explicit, encoding="utf-8")
    print(json.dumps({"profile": name, "directory": str(directory.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
