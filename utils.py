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
import random
import re
import yaml

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


# ========== 背景选项解析与随机组合 ==========
_BACKGROUND_PATH = BASE_DIR / "Prompt" / "background.md"
_WRITING_TIPS_YAML = BASE_DIR / "Prompt" / "writing_tips.yaml"


@lru_cache(maxsize=1)
def load_background_options() -> dict:
    """从 Prompt/background.md 解析可选项。

    预期 Markdown 结构（更可读的格式）：
    ## 时间 / ## 地点 / ## 文明形态 / ## 氛围基调
    - 子类：选项1, 选项2, 选项3

    返回：{"时间": [...], "地点": [...], "文明形态": [...], "氛围基调": [...]}，每类为扁平列表。
    解析失败时，返回内置兜底选项。
    """
    categories = {"时间": [], "地点": [], "文明形态": [], "氛围基调": []}
    current: str | None = None

    try:
        text = _BACKGROUND_PATH.read_text(encoding="utf-8")
    except Exception:
        text = ""

    if text:
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            # 识别二级标题
            if line.startswith("## "):
                title = line[3:].strip()
                # 标题里可能含有括号或英文
                for key in categories.keys():
                    if key in title:
                        current = key
                        break
                else:
                    current = None
                continue

            # 项目行：- 子类：选项1, 选项2 / 也兼容 "•" 开头
            if current and (line.startswith("- ") or line.startswith("• ")):
                content = line[2:].strip()
                # 去掉可能的 markdown 行内强调
                content = content.strip("*_")
                # 子类与选项用冒号分隔
                parts = re.split(r"[：:]", content, maxsplit=1)
                opts_part = parts[1] if len(parts) > 1 else parts[0]
                # 将不同分隔符统一拆分：中文逗号、顿号、斜杠、英文逗号、分号
                options = [
                    o.strip()
                    for o in re.split(r"[、，,/;]|\\s*/\\s*", opts_part)
                    if o.strip()
                ]
                # 去重保持顺序
                seen = set(categories[current])
                for o in options:
                    if o not in seen:
                        categories[current].append(o)
                        seen.add(o)

    # 兜底：若某一类仍为空，用内置默认
    if not any(categories.values()) or any(len(v) == 0 for v in categories.values()):
        categories = {
            "时间": ["上古神话", "秦汉", "中世纪", "维多利亚时期", "二战", "现代都市", "近未来（22 世纪）", "星际时代", "末日余生", "虚构纪元"],
            "地点": ["纽约", "伊斯坦布尔", "上海", "东京", "中土世界", "星际殖民地", "魔法大陆", "废墟中的伦敦", "赛博朋克北京", "末日后的巴黎", "荒岛", "地堡", "飞船", "豪宅", "地下城"],
            "文明形态": ["普通社会", "王朝", "帝国", "部落", "AI 主宰", "星际帝国", "赛博朋克城市", "魔法王国", "精灵与矮人", "神明干预", "废土反乌托邦", "鬼怪横行", "吸血鬼社会", "异次元裂缝", "奴隶社会", "极端宗教", "乌托邦实验地"],
            "氛围基调": ["浪漫田园", "都市现代", "荒凉末日", "密闭悬疑", "奇诡魔幻", "黑暗恐怖"],
        }

    return categories


def random_background_text() -> str:
    """随机组合一个小说背景（时间/地点/文明形态/氛围基调）。"""
    cats = load_background_options()
    try:
        time_opt = random.choice(cats.get("时间", []) or ["未来（22 世纪）"])
        place_opt = random.choice(cats.get("地点", []) or ["末日后的伊斯坦布尔"])
        civ_opt = random.choice(cats.get("文明形态", []) or ["废土反乌托邦"])
        mood_opt = random.choice(cats.get("氛围基调", []) or ["荒凉末日"])
    except IndexError:
        # 极端情况下的兜底
        time_opt, place_opt, civ_opt, mood_opt = "未来（22 世纪）", "末日后的伊斯坦布尔", "废土反乌托邦", "荒凉末日"

    return (
        f"时间：{time_opt}\n"
        f"地点：{place_opt}\n"
        f"文明形态：{civ_opt}\n"
        f"氛围基调：{mood_opt}"
    )


# ========== 写作 Tips (YAML) 读取与抽取 ==========
@lru_cache(maxsize=1)
def load_writing_tips() -> dict:
    """加载 Prompt/writing_tips.yaml，返回标准化结构。

    期望结构：
    {
      'version': int,
      'updated': 'YYYY-MM-DD' | str,
      'categories': {
         '<key>': {'name': str, 'tips': [str, ...]},
         ...
      }
    }
    """
    try:
        text = _WRITING_TIPS_YAML.read_text(encoding="utf-8")
        data = yaml.safe_load(text) or {}
        # 轻度校验
        if not isinstance(data, dict) or "categories" not in data:
            raise ValueError("invalid writing_tips.yaml: missing categories")
        cats = data.get("categories", {})
        if not isinstance(cats, dict) or not cats:
            raise ValueError("invalid writing_tips.yaml: empty categories")
        # 清洗：确保每个分类都有 tips 列表
        norm_cats: dict[str, dict] = {}
        for k, v in cats.items():
            if not isinstance(v, dict):
                continue
            name = v.get("name") or k
            tips = v.get("tips") or []
            if isinstance(tips, list):
                tips = [str(x).strip() for x in tips if str(x).strip()]
            else:
                tips = []
            if tips:
                norm_cats[k] = {"name": str(name), "tips": tips}
        data["categories"] = norm_cats
        return data
    except Exception:
        # 兜底，返回空结构，避免崩溃
        return {"version": 0, "updated": "", "categories": {}}


def list_tip_categories() -> list[tuple[str, str, int]]:
    """列出可用分类：[(key, name, tip_count), ...]"""
    data = load_writing_tips()
    out: list[tuple[str, str, int]] = []
    for k, v in data.get("categories", {}).items():
        out.append((k, v.get("name", k), len(v.get("tips", []))))
    # 稳定排序：按 key
    out.sort(key=lambda x: x[0])
    return out


def random_tip(category_key: str | None = None) -> tuple[str, str]:
    """抽取一个 tip。

    返回 (category_name, tip)。当 category_key 无效或为空时，从所有分类中随机抽取。
    若无可用数据，返回 ("", "")。
    """
    data = load_writing_tips()
    cats = data.get("categories", {})
    if not cats:
        return "", ""

    if category_key and category_key in cats:
        c = cats[category_key]
        tips = c.get("tips", [])
        if tips:
            return c.get("name", category_key), random.choice(tips)

    # 混合所有 tip
    pool: list[tuple[str, str]] = []
    for k, v in cats.items():
        cname = v.get("name", k)
        for t in v.get("tips", []) or []:
            pool.append((cname, t))
    if not pool:
        return "", ""
    return random.choice(pool)


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


def build_outline_messages(background: str, role: str,
                           decisions: List[Tuple[str, str]],
                           date_cn: str) -> List[Dict[str, str]]:
    """构建【故事大纲】生成的消息。

    要求：
    - 结合背景、主人公与已做的抉择。
    - 用英文思考并输出一个清晰的故事大纲（建议三幕式，含意外反转或余味）。
    - 然后给出该英文大纲的中文翻译。
    输出示例结构：
    English Outline:\n...
    中文翻译：\n...
    """
    decisions_lines = "\n".join([f"- {q} → {a}" for q, a in decisions])
    user_cn = (
        "请基于以下信息先生成故事大纲：\n"
        f"故事背景：{background}\n"
        f"主人公：{role}\n"
        f"提示与抉择：\n{decisions_lines}\n\n"
        "要求：\n"
        "- Think in English and write a concise, three-act style story outline, with an unexpected twist or lingering aftertaste at the end.\n"
        "- After the English outline, provide a faithful Chinese translation.\n"
        "- Keep both parts compact and focused on plot, character motivation, setting, and key beats.\n\n"
        "请严格按以下格式输出：\n"
        "English Outline:\n[英文大纲]\n\n中文翻译：\n[中文大纲]"
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a sharp, imaginative story architect. You think clearly in English first, structure arcs and beats, and present bilingual outlines."
            ),
        },
        {"role": "user", "content": user_cn},
    ]


def build_novel_messages(outline_text: str, style: str) -> List[Dict[str, str]]:
    """构建【根据大纲写小说】的消息。

    参数：
    - outline_text: 上一步产出的英文大纲及其中文翻译（原样传入）。
    - style: "经典文体" 或 "黑暗惊悚" 等风格选项。
    行为：
    - 仅根据给定大纲与风格提示，创作中文短篇小说（≤1500字）。
    - 只输出小说内容，不要附加说明或再次输出大纲。
    """
    style_hint = (
        _get_horror_style_hint()
        if style == "黑暗惊悚"
        else "以“经典文体”写作：带读者去看见而非告知，明快却不直白，节奏清晰可朗读，避免术语与套话，让文字像窗户般自然开启。"
    )
    user_cn = (
        "下面是故事大纲（含中英两部分，供参考）：\n"
        f"{outline_text}\n\n"
        "请基于该大纲，用中文写成一篇短篇小说（不超过1500字）。\n"
        "注意：\n"
        "- 不要复述或再次输出大纲；只输出小说正文。\n"
        f"小说风格提示：\n{style_hint}\n"
    )
    return [
        {
            "role": "system",
            "content": (
                "你是一个锐利、有想象力的小说作家。"
            ),
        },
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

    # 基于层级的 tips 引导
    tips_block = ""
    extra_guide = ""
    try:
        if depth == 1:
            # 随机挑选 3 个不同类别，各抽 1 个 tip
            cats = [k for (k, _name, cnt) in list_tip_categories() if cnt > 0]
            random.shuffle(cats)
            picked = cats[:3]
            tips_list: List[str] = []
            for ck in picked:
                cname, tip = random_tip(ck)
                if cname and tip:
                    tips_list.append(f"- [{cname}] {tip}")
            if tips_list:
                tips_block = "写作提示（请从中挑选1个作为主要目标）：\n" + "\n".join(tips_list)
                extra_guide = (
                    "请基于上述三个 tip，并结合背景与人物，先在心中挑选其中一个作为“故事的主要目标”，"
                    "据此提出一个二选一的问题。"
                )
        elif depth == 2:
            cname, tip = random_tip(None)
            if tip:
                tips_block = f"写作提示：\n- [{cname}] {tip}"
                extra_guide = (
                    "参考该 tip，为此前故事设计一个情境；在这个情境处，故事出现关键转折。"
                    "围绕这一转折，提出一个二选一的问题。"
                )
        elif depth == 3:
            cname, tip = random_tip(None)
            if tip:
                tips_block = f"写作提示：\n- [{cname}] {tip}"
                extra_guide = (
                    "从该 tip 中抽取一个独特元素，作为“催化剂”加入故事；"
                    "围绕该催化剂的影响，提出一个二选一的问题。"
                )
    except Exception:
        # tips 获取失败时静默忽略
        tips_block = ""
        extra_guide = ""

    user_cn_parts = [
        f"背景：{background}",
        f"角色：{role}",
        f"当前是第{depth}层抉择。{choices_text}".rstrip(),
    ]
    if tips_block:
        user_cn_parts.append(tips_block)
    # 基本要求
    user_cn_parts.append("请生成一个二选一的问题，与背景和此前的抉择存在逻辑关系。")
    if extra_guide:
        user_cn_parts.append(extra_guide)
    user_cn_parts.append("被用户选中的问题，将作为故事板，供后续写作参考。")
    user_cn_parts.append(
        "请按以下格式返回：\n"
        "问题：[你的问题]\n"
        "True选项：[True选项的极简描述]\n"
        "False选项：[False选项的极简描述]"
    )
    user_cn = "\n".join(user_cn_parts)
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