# -*- coding: utf-8 -*-
"""
CUHKSZ 提问模块
处理CUHKSZ相关的问答功能
"""

from typing import Dict
import glob
from pathlib import Path
import streamlit as st

from utils import BASE_DIR, generate_chat


PROMPT_DIR = BASE_DIR / "Prompt"   # 用于 CUHKSZ 分支读取模板


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
    """CUHKSZ 问答模式主页面"""
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