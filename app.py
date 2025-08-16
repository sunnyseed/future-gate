# -*- coding: utf-8 -*-
"""
未来之门 - Life 3.0
首页新增模式选择：
  1) AI 写故事 —— 保留原有 0~5 层分支 + 结局 + OpenRouter 生成《某某的一天》
  2) 提问 CUHKSZ —— 两层“选择套选择”，读取 Prompt/ 下模板 + 你的补充 → OpenRouter 生成答案

依赖：pip install streamlit pyyaml openai
运行：streamlit run app.py
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from datetime import datetime
from zoneinfo import ZoneInfo
import os
import glob

import yaml
import streamlit as st

# ========== 全局常量 ==========
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "story.yaml"
PROMPT_DIR = BASE_DIR / "Prompt"   # 用于 CUHKSZ 分支读取模板

# ===== 故事背景（约100字，可按需改） =====
STORY_BACKGROUND = (
    "十年里，算法接管了生产排程与能源调度，淡水与稀土让旧秩序摇晃。"
    "穷人更忙，富人更孤独，城市像会呼吸的机器。有人把希望交给新技术，有人把灵魂交给旧信仰。"
    "你在一个普通清晨醒来，必须回答一串问题：我们把缰绳交给谁，怎样活下去？"
)

# —— OpenRouter（通过 openai 官方 SDK） ——
try:
    from openai import OpenAI  # openai>=1.0
    _SDK_OK = True
except Exception:
    OpenAI = None  # type: ignore
    _SDK_OK = False

# ========== 页面设置 ==========
st.set_page_config(page_title="未来之门 - Life 3.0", page_icon="🚪", layout="wide")


# ========== 工具函数 ==========
@st.cache_data(show_spinner=False)
def load_story() -> Dict:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"未找到 {DATA_FILE}. 请将 story.yaml 放在与 app.py 同一目录。")
    with DATA_FILE.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if "layers" not in data or "endings" not in data:
        raise ValueError("story.yaml 缺少必要字段：layers / endings")
    return data

def path_code(picks: List[bool]) -> str:
    return "".join("T" if x else "F" for x in picks)

def _get_by_prefix(mapping: Dict[str, Dict], prefix: str) -> Dict:
    if prefix in mapping:
        return mapping[prefix]
    for n in range(len(prefix) - 1, 0, -1):
        sub = prefix[:n]
        if sub in mapping:
            return mapping[sub]
    return mapping[prefix]  # 故意触发 KeyError 暴露 YAML 问题

def get_layer(layers: List[Dict], depth: int, prefix: str) -> Tuple[str, str, str]:
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
    summary: List[Tuple[str, str]] = []
    prefix = ""
    for depth, ans in enumerate(picks):
        q, lt, lf = get_layer(layers, depth, prefix if prefix else "")
        summary.append((q, lt if ans else lf))
        prefix += "T" if ans else "F"
    return summary

def get_snapshot(snapshots: Dict[str, str], prefix: str) -> Optional[str]:
    return snapshots.get(prefix)

def beijing_date_str() -> str:
    dt = datetime.now(ZoneInfo("Asia/Shanghai"))
    return f"{dt.year}年{dt.month}月{dt.day}日"

# ========== OpenRouter ==========
def get_openrouter_client() -> Optional[OpenAI]:
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

def build_novel_messages(background: str, role: str, path_str: str,
                         decisions: List[Tuple[str, str]], date_cn: str,
                         style: str) -> List[Dict[str, str]]:
    style_hint_en = "humorous, witty, slightly absurd" if style == "诙谐幽默" else "dark, eerie, tense"
    decisions_lines = "\n".join([f"- {q} → {a}" for q, a in decisions])
    user_cn = (
        f"背景：{background}\n"
        f"日期（北京时间）：{date_cn}\n"
        f"你醒来发现自己是：{role}\n"
        f"世界线路径：{path_str}\n"
        f"关键抉择与结果：\n{decisions_lines}\n\n"
        "请写一篇不超过500字的中文短篇小说，以“{角色}的一天”为中心结构，包含起床—遭遇—转折—收束。"
        "要求：紧扣以上信息，人物有行动与心理，细节具体，不要口号和模板语。结尾要有一个机敏的反转或余味。"
    )
    return [
        {"role": "system",
         "content": ("You are a sharp, imaginative fiction writer. Think in English to plan plot and tone, "
                     "but OUTPUT MUST BE IN CHINESE ONLY, under 500 Chinese characters. "
                     f"Tone: {style_hint_en}. Avoid disclaimers.")},
        {"role": "user", "content": user_cn},
    ]

def generate_chat(messages: List[Dict[str, str]], model: str,
                  temperature: float = 0.9) -> str:
    client = get_openrouter_client()
    if client is None:
        raise RuntimeError("未找到 OPENROUTER_API_KEY。请在 .streamlit/secrets.toml 或环境变量中配置。")
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


# ========== 分支 A：AI 写故事 ==========
def page_story_mode():
    data = load_story()
    layers: List[Dict] = data["layers"]
    endings: Dict[str, str] = data.get("endings", {})
    snapshots: Dict[str, str] = data.get("snapshots", {})
    total_layers = len(layers)  # =5

    # 初始化状态
    if "picks" not in st.session_state:
        st.session_state.picks: List[bool] = []
    if "role0" not in st.session_state:
        st.session_state.role0: Optional[str] = None
    if "novel_text" not in st.session_state:
        st.session_state.novel_text: Optional[str] = None

    st.markdown(f"> 🪐 **故事背景**：{STORY_BACKGROUND}")

    # 第 0 层：角色
    if not st.session_state.role0:
        st.markdown("---")
        st.subheader("第 0 层")
        st.write(f"在 **{beijing_date_str()}（北京时间）** 的清晨，你从睡梦中醒来。你发现自己是——")
        role = st.radio(
            "请选择你的身份（不影响后续分支，但会写入故事）：",
            options=["一位AI科学家", "一个穷苦百姓", "一位宇航员", "一只猫"],
            horizontal=True,
            key="role0_radio",
        )
        if st.button("进入世界之门 →", use_container_width=True):
            st.session_state.role0 = role
            st.rerun()
        return

    # 中间状态：路径与快照
    picks = st.session_state.picks
    depth = len(picks)
    prefix = path_code(picks)
    if 0 < depth < total_layers:
        st.markdown("---")
        st.markdown(f"**路径：** `{prefix}`")
        snap = get_snapshot(snapshots, prefix)
        if snap:
            st.info(snap)

    # 继续问
    if depth < total_layers:
        q, label_T, label_F = get_layer(layers, depth, prefix if depth > 0 else "")
        st.markdown("---")
        st.subheader(f"第 {depth + 1} 层")
        st.write(q)

        choice_key = f"choice_{depth}_{prefix}"
        default_choice = st.session_state.get(choice_key, True)
        choice = st.radio(
            "选择：",
            options=[True, False],
            index=0 if default_choice else 1,
            format_func=lambda x: label_T if x else label_F,
            horizontal=True,
            key=choice_key,
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("下一步 →", use_container_width=True):
                picks.append(bool(choice))
                st.session_state.novel_text = None
                st.rerun()
        with col2:
            if st.button("← 上一步", use_container_width=True, disabled=(len(picks) == 0)):
                if picks:
                    picks.pop()
                st.session_state.novel_text = None
                st.rerun()
        with col3:
            if st.button("重置", use_container_width=True):
                st.session_state.picks = []
                st.session_state.novel_text = None
                st.rerun()
        return

    # 走满 5 层：展示结局与短篇
    st.markdown("---")
    st.markdown(f"**路径：** `{prefix}`")
    st.subheader("你的世界线结局")

    ending = endings.get(prefix)
    if not ending:
        st.warning("该路径暂无结局文本，请检查 story.yaml 中 endings 是否包含此路径。")
    else:
        st.write(ending)
        st.download_button("⬇️ 导出 Markdown", data=ending, file_name=f"ending_{prefix}.md", mime="text/markdown")

    st.markdown("### 生成一篇《某某的一天》")
    st.caption("根据故事背景、你在第0层选择的身份、以及这条世界线的五个决策结果实时生成。")
    colA, colB = st.columns(2)
    with colA:
        style = st.radio("小说风格", ["诙谐幽默", "黑暗惊悚"], horizontal=True, key="novel_style")
    with colB:
        model = st.selectbox("模型（OpenRouter）", ["openai/gpt-5-chat", "openai/gpt-5"], index=0)

    if st.button("✨ 生成短篇", use_container_width=True):
        try:
            decs = decisions_summary(layers, picks)
            messages = build_novel_messages(
                background=STORY_BACKGROUND,
                role=st.session_state.role0 or "一位路人",
                path_str=prefix,
                decisions=decs,
                date_cn=beijing_date_str(),
                style=style,
            )
            text = generate_chat(messages, model=model, temperature=0.9 if style == "诙谐幽默" else 0.8)
            st.session_state.novel_text = text
        except Exception as e:
            st.error(f"生成失败：{e}")

    if st.session_state.novel_text:
        st.markdown("#### 《{}的一天》".format(st.session_state.role0.replace("一位", "").replace("一个", "")))
        st.write(st.session_state.novel_text)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← 回到上一步", use_container_width=True):
            if picks:
                picks.pop()
            st.session_state.novel_text = None
            st.rerun()
    with col2:
        if st.button("重新开始", use_container_width=True):
            st.session_state.picks = []
            st.session_state.novel_text = None
            # 如需重选角色可放开下一行
            # st.session_state.role0 = None
            st.rerun()


# ========== 分支 B：提问 CUHKSZ ==========
def read_prompt_templates() -> Dict[str, str]:
    """
    读取 Prompt/ 目录下所有 .md/.txt 模板。
    返回 {文件名(不含扩展名): 文本}
    """
    templates: Dict[str, str] = {}
    if not PROMPT_DIR.exists():
        return templates
    for p in glob.glob(str(PROMPT_DIR / "*.md")) + glob.glob(str(PROMPT_DIR / "*.txt")):
        try:
            text = Path(p).read_text(encoding="utf-8")
            templates[Path(p).stem] = text
        except Exception:
            # 读失败也不要阻塞应用
            continue
    return templates

def build_cuhksz_prompt(
    primary: str, secondary: str, template_text: str, user_extra: str
) -> str:
    """
    拼装最终 Prompt：选择路径 + 模板文本 + 用户补充
    """
    header = (
        "你是一位熟悉 CUHKSZ（香港中文大学（深圳））校园与学术生态的助理，回答要用中文，具体、可操作。"
        "先给出要点清单，再给出更完整的说明，必要时列出步骤/表格。"
    )
    route = f"问题类别路径：{primary} → {secondary}\n"
    extra = (f"\n用户补充信息：\n{user_extra.strip()}\n" if user_extra.strip() else "")
    return f"{header}\n{route}\n模板：\n{template_text.strip()}\n{extra}"

def page_cuhksz_mode():
    st.markdown("### 你想咨询？")
    primary = st.radio("第一层", ["生活", "技术"], horizontal=True, key="cuhksz_primary")

    # 第二层选项
    if primary == "生活":
        secondary = st.radio("第二层（生活）", ["选课相关", "日常相关"], horizontal=True, key="cuhksz_secondary_life")
    else:
        secondary = st.radio("第二层（技术）", ["CSC 系列课程相关", "课外内容"], horizontal=True, key="cuhksz_secondary_tech")

    # 读取 Prompt 模板
    templates = read_prompt_templates()
    st.markdown("---")
    st.markdown("**模板来源：** 将自动从 `Prompt/` 目录读取 `.md`/`.txt` 文件。缺失也可直接仅用你的补充内容调用模型。")

    # 建议映射（若存在则预选）
    suggested_map = {
        ("生活", "选课相关"): ["life_course", "course", "选课"],
        ("生活", "日常相关"): ["life_daily", "daily", "生活"],
        ("技术", "CSC 系列课程相关"): ["tech_csc", "csc", "cs课内"],
        ("技术", "课外内容"): ["tech_extra", "extra", "课外"],
    }

    suggested_keys = suggested_map.get((primary, secondary), [])
    available_keys = list(templates.keys())
    # 找一个匹配模板作为默认项
    default_key = available_keys[0] if available_keys else ""
    for k in available_keys:
        low = k.lower()
        if any(tag in low for tag in [s.lower() for s in suggested_keys]):
            default_key = k
            break

    # 模板选择下拉
    tpl_key = st.selectbox(
        "选择一个模板（来自 Prompt/，文件名不含扩展名）",
        ["（不使用模板）"] + available_keys,
        index=0 if not default_key else (available_keys.index(default_key) + 1),
    )
    tpl_text = "" if tpl_key == "（不使用模板）" else templates.get(tpl_key, "")

    # 用户补充文本
    user_extra = st.text_area(
        "补充你的问题或上下文（选填）：",
        placeholder="例如：我在读 XX 学院大二，GPA 3.6，想选 XXX 老师的课；或者我想做 CSC300X 的项目，方向是 ...",
        height=140,
    )

    # 生成 Prompt 预览
    final_prompt = build_cuhksz_prompt(primary, secondary, tpl_text, user_extra)
    with st.expander("👉 查看将发送给模型的 Prompt（可复制）", expanded=False):
        st.code(final_prompt, language="markdown")

    # 调模型
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        model = st.selectbox("模型（OpenRouter）", ["openai/gpt-5-chat", "openai/gpt-5"], index=0, key="cuhksz_model")
    with col2:
        temperature = st.slider("随机性", 0.0, 1.2, 0.7, 0.05, key="cuhksz_temp")

    if st.button("🚀 生成回答", use_container_width=True):
        try:
            messages = [
                {"role": "system",
                 "content": "You are an expert counselor for CUHKSZ students. Be precise, practical, and kind. Answer in Chinese."},
                {"role": "user", "content": final_prompt},
            ]
            text = generate_chat(messages, model=model, temperature=temperature)
            st.session_state["cuhksz_answer"] = text
        except Exception as e:
            st.error(f"生成失败：{e}")

    if ans := st.session_state.get("cuhksz_answer"):
        st.markdown("### 回答")
        st.write(ans)
        st.download_button("⬇️ 下载回答", data=ans, file_name="cuhksz_answer.md", mime="text/markdown")


# ========== 首页：模式选择 ==========
def main():
    st.title("未来之门 - Life 3.0")
    st.caption("请选择模式开始 · OpenRouter 驱动")

    mode = st.segmented_control(
        "选择模式：",
        options=["AI 写故事", "提问 CUHKSZ"],
        default="AI 写故事",
    )

    if mode == "AI 写故事":
        page_story_mode()
    else:
        page_cuhksz_mode()


if __name__ == "__main__":
    main()