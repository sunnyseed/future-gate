# -*- coding: utf-8 -*-
"""
未来之门 - Life 3.0
首页新增模式选择：
    1) AI 写故事动态版
    2) 提问 CUHKSZ

依赖：pip install streamlit pyyaml openai
运行：streamlit run app.py
"""

import streamlit as st
from ai_story import page_dynamic_story_mode
from cuhksz_mode import page_cuhksz_mode

# ========== 页面设置 ==========
st.set_page_config(page_title="未来之门 - Life 3.0", page_icon="🚪", layout="wide")


# ========== 首页：模式选择 ==========
def main():
    st.title("未来之门 - Life 3.0")
    st.caption("请选择模式开始 · OpenRouter 驱动")

    mode = st.segmented_control(
        "选择模式：",
        options=["提问 CUHKSZ", "AI 写故事动态版"],
        default="AI 写故事动态版",
    )

    if mode == "提问 CUHKSZ":
        page_cuhksz_mode()
    else:
        page_dynamic_story_mode()


if __name__ == "__main__":
    main()