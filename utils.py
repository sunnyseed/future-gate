# -*- coding: utf-8 -*-
"""
共享工具模块
包含所有模块共用的常量、工具函数和配置
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Iterator
from datetime import datetime
from zoneinfo import ZoneInfo
import os
import json
from functools import lru_cache

import streamlit as st

# ========== 全局常量 ==========
BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
try:
    LOG_DIR.mkdir(exist_ok=True)
except Exception:
    pass

# ===== 故事背景（可按需改） =====
STORY_BACKGROUND = (
    "你是一个小说作家，为客户定制一部 废土世界背景 虚构小说。\n"
    "时间：22 世纪初\n"
    "地点：伊斯坦布尔\n"
    "人类–变种人大战争后，人类惨胜，旧世界已崩溃，科技和文明失落，奉行丛林法则。变种人遭到猎杀，出生即成为奴隶。存在辐射飓风、尸鬼等异常现象。"
)

# （已移除写作规则 WRITING_RULES）

# ===== 出题模型选项（用于动态问题生成） =====
QUESTION_MODEL_OPTIONS = [
    "openai/o3",
    "openai/gpt-5-chat",
    "openai/gpt-5",
    "deepseek/deepseek-r1-0528",
    "google/gemini-2.5-pro",
    "anthropic/claude-opus-4.1",
]

# —— OpenRouter（通过 openai 官方 SDK） ——
try:
    from openai import OpenAI  # openai>=1.0
    _SDK_OK = True
except Exception:
    OpenAI = None  # type: ignore
    _SDK_OK = False


# ========== 工具函数 ==========
def path_code(picks: List[bool]) -> str:
    """将选择列表转换为路径码"""
    return "".join("T" if x else "F" for x in picks)


def _get_by_prefix(mapping: Dict[str, Dict], prefix: str) -> Dict:
    """根据前缀获取配置"""
    if prefix in mapping:
        return mapping[prefix]
    for n in range(len(prefix) - 1, 0, -1):
        sub = prefix[:n]
        if sub in mapping:
            return mapping[sub]
    return mapping[prefix]  # 故意触发 KeyError 暴露 YAML 问题


def get_layer(layers: List[Dict], depth: int, prefix: str) -> Tuple[str, str, str]:
    """获取指定层的问题和选项"""
    layer = layers[depth]
    if "question" in layer:
        q = layer["question"]
        opts = layer["options"]
        return q, opts["T"], opts["F"]
    q_map = layer.get("question_by_path", {})
    o_map = layer.get("options_by_path", {})
    q_entry = _get_by_prefix(q_map, prefix)
    o_entry = _get_by_prefix(o_map, prefix)
    return q_entry, o_entry["T"], o_entry["F"]


def decisions_summary(layers: List[Dict], picks: List[bool]) -> List[Tuple[str, str]]:
    """生成决策摘要"""
    summary: List[Tuple[str, str]] = []
    prefix = ""
    for depth, ans in enumerate(picks):
        q, lt, lf = get_layer(layers, depth, prefix if prefix else "")
        summary.append((q, lt if ans else lf))
        prefix += "T" if ans else "F"
    return summary


def get_snapshot(snapshots: Dict[str, str], prefix: str) -> Optional[str]:
    """获取路径快照"""
    return snapshots.get(prefix)


def beijing_date_str() -> str:
    """获取北京时间日期字符串"""
    dt = datetime.now(ZoneInfo("Asia/Shanghai"))
    return f"{dt.year}年{dt.month}月{dt.day}日"


def write_log(event: str, data: dict | None = None, session_id: str | None = None) -> None:
    """将事件以 JSON 行写入日志文件（按天滚动）。静默失败。"""
    try:
        dt = datetime.now(ZoneInfo("Asia/Shanghai"))
        log_path = LOG_DIR / f"story_{dt.strftime('%Y%m%d')}.log"
        record = {
            "ts": dt.isoformat(),
            "event": event,
        }
        if session_id:
            record["session"] = session_id
        if data is not None:
            record["data"] = data
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # 避免日志失败影响主流程
        pass


# ========== OpenRouter ==========
def get_openrouter_client() -> Optional[object]:
    """获取OpenRouter客户端"""
    if not _SDK_OK:
        return None
    
    # 读取 Key：secrets 优先，环境变量兜底
    or_key = ""
    try:
        or_key = st.secrets.get("OPENROUTER_API_KEY", "")
    except Exception:
        or_key = ""
    if not or_key:
        or_key = os.getenv("OPENROUTER_API_KEY", "")
    if not or_key:
        return None

    # 头部只用 ASCII，避免 'ascii' codec 报错
    headers = {
        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost:8501"),
        "X-Title": "FutureGate-Life3",
    }
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=or_key,
        default_headers=headers,
    )
    return client


@lru_cache(maxsize=1)
def _get_horror_style_hint() -> str:
    """从 Prompt/horror_fiction.md 读取风格提示，带缓存和兜底。"""
    try:
        p = BASE_DIR / "Prompt" / "horror_fiction.md"
        text = p.read_text(encoding="utf-8").strip()
        # 若文件存在但内容为空，仍然回退到默认提示
        return text if text else "dark, eerie, tense"
    except Exception:
        # 兜底，避免运行时崩溃
        return "dark, eerie, tense"


def build_novel_messages(background: str, role: str, path_str: str,
                         decisions: List[Tuple[str, str]], date_cn: str,
                         style: str) -> List[Dict[str, str]]:
    """构建小说生成的消息"""
    style_hint = "humorous, witty, slightly absurd" if style == "诙谐幽默" else _get_horror_style_hint()
    decisions_lines = "\n".join([f"- {q} → {a}" for q, a in decisions])
    user_cn = (
        "参考下面的提示:\n"
        f"故事背景：{background}\n"
        f"主人公：{role}\n"
        f"提示与抉择：\n{decisions_lines}\n\n"
        "先设计自行一个三幕式的故事大纲，之后写成短篇小说,结尾需要出现意外反转或余味，不超过 1500 字。\n"
        "只需要回复小锁文内容，不要多余说明。\n\n"
        f"小说风格提示：\n{style_hint}\n\n"
    )
    return [
        {"role": "system",
         "content": ("你是一个锐利、有想象力的小说作家。")},
        {"role": "user", "content": user_cn},
    ]


def generate_chat(messages: List[Dict[str, str]], model: str,
                  temperature: float = 0.9) -> str:
    """调用OpenRouter生成聊天回复"""
    client = get_openrouter_client()
    if client is None:
        raise RuntimeError("未找到 OPENROUTER_API_KEY。请在 .streamlit/secrets.toml 或环境变量中配置。")
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


def generate_chat_stream(messages: List[Dict[str, str]], model: str,
                         temperature: float = 0.9) -> Iterator[str]:
    """流式生成聊天回复，逐段产出文本。"""
    client = get_openrouter_client()
    if client is None:
        raise RuntimeError("未找到 OPENROUTER_API_KEY。请在 .streamlit/secrets.toml 或环境变量中配置。")
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,
    )
    # 逐块产出内容
    for chunk in stream:  # type: ignore[assignment]
        try:
            choice = chunk.choices[0]
            delta = getattr(choice, "delta", None)
            if delta and getattr(delta, "content", None):
                yield delta.content  # type: ignore[attr-defined]
        except Exception:
            # 忽略无法解析的块（如role变更等）
            continue


def build_dynamic_roles_messages(background: str, date_cn: str) -> List[Dict[str, str]]:
    """生成动态角色选择的提示词"""
    user_cn = (
        f"背景：{background}\n"
        "请根据这个背景，生成4个不同的角色供用户选择。每个角色都应该有独特的身份和背景，至少有一个动物角色"
        "能够在这个世界中展开有趣的故事。角色描述要简洁但生动，每个不超过15字。"
        "请直接返回4个角色，每行一个，不要编号或其他格式。"
    )
    return [
        {"role": "system", "content": "You are a creative storyteller. Generate 4 unique character roles based on the given background. Output in Chinese only, one role per line, no numbering."},
        {"role": "user", "content": user_cn},
    ]


def build_dynamic_question_messages(background: str, role: str, depth: int, 
                                   previous_choices: List[Tuple[str, str]], date_cn: str) -> List[Dict[str, str]]:
    """生成动态问题的提示词"""
    choices_text = ""
    if previous_choices:
        choices_lines = "\n".join([f"- {q} → {a}" for q, a in previous_choices])
        choices_text = f"\n之前的抉择：\n{choices_lines}\n"
    
    user_cn = (
        f"背景：{background}\n"
        f"角色：{role}\n"
        f"当前是第{depth}层抉择。{choices_text}\n\n"
        "请生成一个二选一的问题，问题涉及用户对情节和人物的喜好，将是作为故事板，供后续写作参考。\n"
        "1. 与背景和此前的抉择有逻辑关系。\n"
        "2. 可以是关于主人公的一个欲望、主要的配角、其潜意识的矛盾、角色的性格、实现欲望的手段等方面的线索。\n"
        "请按以下格式返回：\n"
        "问题：[你的问题]\n"
        "True选项：[True选项的极简描述]\n"
        "False选项：[False选项的极简描述]"
    )
    return [
        {"role": "system", "content": "You are a creative storyteller. Generate a True/False question with options. Think in English, output in Chinese only, follow the exact format specified."},
        {"role": "user", "content": user_cn},
    ]


def parse_dynamic_response(response: str) -> Tuple[str, str, str]:
    """解析动态生成的响应，提取问题、True选项和False选项"""
    lines = response.strip().split('\n')
    question = ""
    true_option = ""
    false_option = ""
    
    for line in lines:
        line = line.strip()
        if line.startswith("问题："):
            question = line[3:].strip()
        elif line.startswith("True选项："):
            true_option = line[len("True选项："):].strip()
        elif line.startswith("False选项："):
            false_option = line[len("False选项："):].strip()
    
    return question, true_option, false_option