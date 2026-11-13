"""从固定来源获取带校验的公开摄影样本，不生成模拟检测指标。"""

import argparse
import json
import os
import shutil
from pathlib import Path

import pooch
from PIL import Image

SOURCES = {
    "astronaut.png": "88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5",
    "chelsea.png": "596aa1e7cb875eb79f437e310381d26b338a81c2da23439704a73c4651e8c4bb",
    "coffee.png": "cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7",
    "rocket.jpg": "c2dd0de7c538df8d111e479619b129464d0269d0ae5fd18ca91d33a7fdfea95c",
}
SOURCE_ROOT = "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.25.2/skimage/data/"


def main() -> None:
    """使用 Pooch 的哈希校验与缓存生成可复现的示例导入材料。

    :returns: 无；图片写入 data，来源与分组说明写入 docs/examples。
    """
    if not os.getenv("CONDA_PREFIX") or os.getenv("CONDA_DEFAULT_ENV") == "base":
        raise RuntimeError("请在项目独立 Conda 环境运行")
    parser = argparse.ArgumentParser(description="准备真实摄影样本与重复图片输入")
    parser.add_argument("--include-invalid", action="store_true", help="额外生成一个明确损坏的输入，用于验证隔离能力")
    args = parser.parse_args()
    root = Path("data/example-images")
    docs = Path("docs/examples")
    root.mkdir(parents=True, exist_ok=True)
    docs.mkdir(parents=True, exist_ok=True)
    for name, digest in SOURCES.items():
        pooch.retrieve(url=SOURCE_ROOT + name, known_hash=f"sha256:{digest}", fname=name, path=root, progressbar=False)
    with Image.open(root / "astronaut.png") as image:
        image.convert("RGB").save(root / "astronaut-quality.jpeg", quality=92)
    with Image.open(root / "chelsea.png") as image:
        image.save(root / "chelsea.webp", quality=90)
    shutil.copyfile(root / "astronaut.png", root / "astronaut-copy.png")
    if args.include_invalid:
        (root / "broken.png").write_bytes(b"corrupted-image-verification")
    groups = {
        "astronaut.png": {"video_id": "video-a"}, "astronaut-copy.png": {"video_id": "video-a"},
        "astronaut-quality.jpeg": {"video_id": "video-a"}, "chelsea.png": {"scene_id": "scene-b"},
        "chelsea.webp": {"scene_id": "scene-b"}, "coffee.png": {"batch_id": "batch-c"},
        "rocket.jpg": {"batch_id": "batch-d"},
    }
    (docs / "groups.json").write_text(json.dumps(groups, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sources = {name: {"url": SOURCE_ROOT + name, "sha256": digest} for name, digest in SOURCES.items()}
    (docs / "sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"directory": str(root), "metadata": str(docs / "groups.json"), "sources": str(docs / "sources.json"), "derived_inputs": ["astronaut-copy.png", "astronaut-quality.jpeg", "chelsea.webp"], "is_training_data": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
