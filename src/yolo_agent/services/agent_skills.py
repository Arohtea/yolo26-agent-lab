"""复用 SkillKit 按需读取固定来源的训练知识，不执行技能中的命令。"""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from yolo_agent.services.datasets import DatasetError
from yolo_agent.settings import Settings

SKILL_TOOLS = frozenset({"Skill", "SkillRead"})
SKILL_RESULT_CHARS = 24_000
MAX_DOCUMENT_CHARS = 20_000


class TrainingSkills:
    """保留上游发现、前置信息解析和读取工具，额外限定已封存的知识文件。"""

    def __init__(self, root: Path):
        from langchain_skillkit import SkillKit
        from langchain_skillkit.types import SkillConfig

        self.root = root.resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("schema_version") != 1 or not self.manifest.get("skills"):
            raise ValueError("技能来源清单无效")
        self.configs = {}
        for name, entry in self.manifest["skills"].items():
            for file_name in entry["files"]:
                self._verify(name, file_name)
            config = SkillConfig.from_directory(self.root / "ultralytics" / name)
            if config.name != name:
                raise ValueError("技能名称与来源清单不一致")
            self.configs[name] = config
        self.kit = SkillKit(str(self.root / "ultralytics"))
        self.tools = {tool.name: tool for tool in self.kit.tools}
        for tool in self.tools.values():
            tool.handle_tool_error = False

    def _verify(self, skill_name: str, file_name: str) -> str:
        expected = self.manifest["skills"].get(skill_name, {}).get("files", {}).get(file_name)
        if not expected or Path(file_name).name != file_name or not file_name.endswith(".md"):
            raise ValueError("只能读取来源清单内的技能及 Markdown 参考文件")
        directory = (self.root / "ultralytics" / skill_name).resolve()
        path = (directory / file_name).resolve()
        if not directory.is_relative_to(self.root) or not path.is_relative_to(directory):
            raise ValueError("技能文件超出授权目录")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError("技能文件指纹与来源清单不一致，请核对后更新技能快照")
        return digest

    def catalog(self) -> list[dict]:
        return [{"name": name, "description": config.description,
            "references": [file for file in self.manifest["skills"][name]["files"] if file != "SKILL.md"]}
            for name, config in self.configs.items()]

    def specs(self) -> list[dict]:
        from langchain_core.utils.function_calling import convert_to_openai_tool

        result = []
        for tool in self.tools.values():
            spec = convert_to_openai_tool(tool)
            spec["function"]["parameters"]["properties"]["skill_name"]["enum"] = sorted(self.configs)
            result.append(spec)
        return result

    def read(self, name: str, arguments: dict) -> dict:
        from langchain_core.tools import ToolException

        if name not in self.tools:
            raise ValueError("未知技能工具")
        args = self.tools[name].args_schema.model_validate(arguments).model_dump()
        skill_name = args["skill_name"]
        if skill_name not in self.configs:
            raise ValueError("该技能不在本次封存的训练知识目录中")
        file_name = args.get("file_name", "SKILL.md")
        try:
            digest = self._verify(skill_name, file_name)
            content = self.tools[name].invoke(args)
        except (ToolException, OSError) as error:
            raise ValueError("技能或参考文件不可读取，请检查名称及技能快照") from error
        if not isinstance(content, str) or len(content) > MAX_DOCUMENT_CHARS:
            raise ValueError("技能文档超过单次阅读上限，请由维护者拆分参考文件后更新快照")
        return {"skill_name": skill_name, "file_name": file_name,
            "source": {"repository": self.manifest["repository"], "revision": self.manifest["revision"]},
            "sha256": digest, "knowledge_only": True, "content": content}


@lru_cache(maxsize=4)
def _load(root: str) -> TrainingSkills:
    try:
        return TrainingSkills(Path(root))
    except ImportError as error:
        raise DatasetError(503, "技能依赖未安装，请在项目环境安装 .[agent]") from error
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise DatasetError(503, "训练技能快照缺失或无效，请核对 docs/skills 与来源清单") from error


def training_skills(settings: Settings) -> TrainingSkills | None:
    """只由平台配置选择知识目录，工具调用不能指定路径。"""
    if not settings.agent_skills_enabled:
        return None
    if settings.agent_skills_dir is not None:
        root = settings.agent_skills_dir.resolve()
    else:
        checkout = Path(__file__).resolve().parents[3] / "docs" / "skills"
        root = checkout if checkout.is_dir() else Path(__file__).resolve().parents[1] / "skill_assets"
    return _load(str(root))
