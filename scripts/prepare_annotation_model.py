"""由操作者显式准备固定白名单的模型，Worker 不执行隐式下载。"""

import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"


def main() -> None:
    """下载固定提交的安全权重和处理器，保存来源与文件指纹。

    :returns: 无；权重及来源清单写入独立存储目录。
    """
    if not os.getenv("CONDA_PREFIX") or os.getenv("CONDA_DEFAULT_ENV") == "base":
        raise RuntimeError("请进入项目独立 Conda 环境")
    from yolo_agent.settings import get_settings

    path = get_settings().annotation_model_path
    snapshot_download(
        repo_id=MODEL_ID, revision=REVISION, local_dir=path,
        allow_patterns=["*.json", "vocab.txt", "model.safetensors"],
        max_workers=2,
    )
    hashes = {}
    for file in path.iterdir():
        if file.is_file() and file.name != "provenance.json":
            with file.open("rb") as stream:
                hashes[file.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    (path / "provenance.json").write_text(json.dumps({"model_id":MODEL_ID,"revision":REVISION,"sha256":hashes}, indent=2)+"\n")
    print(json.dumps({"model_id":MODEL_ID,"revision":REVISION,"directory":str(path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
