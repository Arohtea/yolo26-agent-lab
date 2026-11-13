"""通过官方 Transformers 本地权重执行 Grounding DINO。"""

import json
import hashlib
import re
from pathlib import Path

import torch
from PIL import Image, ImageOps
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from yolo_agent.services.runtime import collect_runtime, require_device
from yolo_agent.settings import Settings
from yolo_agent.services.annotators import finalize_candidates, prediction_record


def resolve_phrase(phrase: str, classes: list[dict]) -> dict | None:
    """只接受唯一的完整提示词/逗号别名，不猜测部分词或歧义短语。"""
    def normalized(value):
        words = re.findall(r"[a-z0-9]+", value.casefold())
        if words and words[0] in {"a", "an", "the"}:
            words = words[1:]
        return " ".join(words)

    label = normalized(phrase)
    matches = [category for category in classes if label and label in {normalized(alias) for alias in [category["prompt"], *category["prompt"].split(",")]}]
    return matches[0] if len(matches) == 1 else None


class GroundingDetector:
    """一个任务加载一次模型，合并提示并拒绝歧义类别。"""

    def __init__(self, settings: Settings):
        """仅加载操作者提前准备的安全权重。

        :param settings: 本地模型目录和设备策略。
        :returns: 无。
        """
        self.device = require_device(collect_runtime(settings.device))
        path = settings.annotation_model_path
        self.provenance = json.loads((path / "provenance.json").read_text())
        if self.provenance.get("model_id") != "IDEA-Research/grounding-dino-tiny" or self.provenance.get("revision") != "a2bb814dd30d776dcf7e30523b00659f4f141c71":
            raise RuntimeError("标注模型不在固定版本白名单中")
        for name, expected in self.provenance["sha256"].items():
            if Path(name).name != name:
                raise RuntimeError("模型来源清单包含越界路径")
            with (path / name).open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                    raise RuntimeError(f"本地模型文件指纹不符：{name}")
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(path, local_files_only=True, trust_remote_code=False, use_safetensors=True).to(self.device).eval()
        torch.set_num_threads(min(torch.get_num_threads(), 4))

    def detect(self, path: Path, classes: list[dict], policy: dict) -> list[dict]:
        """对 EXIF 校正后的像素推理，保存所有候选的审核风险标记。

        :param path: 已验证指纹的原始图片。
        :param classes: 固定类别及英文提示词。
        :param policy: 置信度、重复和候选数量限制。
        :returns: 像素检测框及真实置信度，不设置已审核状态。
        """
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        if not classes:
            return []
        caption = " . ".join(category["prompt"].lower() for category in classes) + " ."
        token_count = len(self.processor.tokenizer.encode(caption, add_special_tokens=True))
        if token_count > self.model.config.max_text_len:
            raise ValueError("合并 DINO 提示词超过模型文本上限，请减少类别或缩短提示词；不会静默截断")
        inputs = self.processor(images=image, text=caption, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output = self.model(**inputs)
        prediction = self.processor.post_process_grounded_object_detection(output, inputs.input_ids, threshold=policy["box_threshold"], text_threshold=policy["text_threshold"], target_sizes=[(image.height, image.width)])[0]
        records = []
        for box, score, phrase in zip(prediction["boxes"], prediction["scores"], prediction["text_labels"], strict=True):
            category = resolve_phrase(phrase, classes)
            record = prediction_record(box, float(score.item()), category, image.width, image.height, policy) if category else None
            if record:
                records.append(record)
        return finalize_candidates(records, policy)
