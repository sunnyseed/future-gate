# -*- coding: utf-8 -*-
"""
AI 写故事模块（仅保留：动态故事模式）
通过 URL 添加 ?debug=1 开启调试
"""

from typing import Tuple, List, Optional
import streamlit as st

from utils import (
    beijing_date_str, build_novel_messages, generate_chat,
    build_dynamic_roles_messages, build_dynamic_question_messages,
    parse_dynamic_response, STORY_BACKGROUND,
    QUESTION_MODEL_OPTIONS, write_log, generate_chat_stream,
    random_background_text, build_outline_messages,
)

# 基于 URL 查询参数的调试开关（?debug=1/true/yes/on）
def _is_debug_enabled() -> bool:
    try:
        # Streamlit >= 1.30: st.query_params 存在
        qp = st.query_params  # type: ignore[attr-defined]
        v = qp.get("debug", None)
    except Exception:
        # 兼容旧接口（以防未来回退）
        try:
            qp = st.experimental_get_query_params()
            v = qp.get("debug")
        except Exception:
            v = None
    if v is None:
        return False
    if isinstance(v, list):
        v = v[0] if v else None
    if v is None:
        return False
    v = str(v).lower()
    return v in ("1", "true", "yes", "on")

def page_dynamic_story_mode():
    """动态生成问题和选项的AI写故事模式"""
    
    # 初始化状态
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(int(__import__("time").time()))
    if "dynamic_role" not in st.session_state:
        st.session_state.dynamic_role = None
    if "dynamic_picks" not in st.session_state:
        st.session_state.dynamic_picks = []
    if "dynamic_questions" not in st.session_state:
        st.session_state.dynamic_questions = []  # (question, true_opt, false_opt)
    if "dynamic_novel_text" not in st.session_state:
        st.session_state.dynamic_novel_text = None
    if "dynamic_outline_text" not in st.session_state:
        st.session_state.dynamic_outline_text = None
    if "dynamic_roles_list" not in st.session_state:
        st.session_state.dynamic_roles_list = []
    if "dynamic_early_end" not in st.session_state:
        st.session_state.dynamic_early_end = False
    
    # 侧边栏：模型与风格设置（默认折叠）
    with st.sidebar:
        st.markdown("### 设置")
        # 出题模型
        if "question_model" not in st.session_state:
            st.session_state.question_model = "openai/gpt-5-chat"
        st.session_state.question_model = st.selectbox(
            "问题生成模型",
            QUESTION_MODEL_OPTIONS,
            index=QUESTION_MODEL_OPTIONS.index(st.session_state.question_model) if st.session_state.question_model in QUESTION_MODEL_OPTIONS else 0,
            key="sidebar_question_model",
        )

        # 大纲/短篇生成模型
        if "gen_model" not in st.session_state:
            st.session_state.gen_model = st.session_state.question_model
        st.session_state.gen_model = st.selectbox(
            "大纲/短篇生成模型",
            QUESTION_MODEL_OPTIONS,
            index=QUESTION_MODEL_OPTIONS.index(st.session_state.gen_model) if st.session_state.gen_model in QUESTION_MODEL_OPTIONS else 0,
            key="sidebar_gen_model",
        )

        # 小说风格
        if "dynamic_novel_style" not in st.session_state:
            st.session_state.dynamic_novel_style = "经典文体"
        st.session_state.dynamic_novel_style = st.radio(
            "小说风格",
            ["经典文体", "黑暗惊悚"],
            horizontal=False,
            key="sidebar_novel_style",
        )
    
    # 调试信息：显示初始化状态
    if _is_debug_enabled():
        st.write(f"初始化状态 - 提前结束: {st.session_state.dynamic_early_end}")
    
    # 状态同步检查：确保picks和questions长度一致
    if len(st.session_state.dynamic_picks) > len(st.session_state.dynamic_questions):
        # 如果picks比questions多，说明状态不同步，重置picks
        st.session_state.dynamic_picks = st.session_state.dynamic_picks[:len(st.session_state.dynamic_questions)]

    # 可编辑的故事背景
    if "custom_background" not in st.session_state:
        st.session_state.custom_background = STORY_BACKGROUND
    
    st.markdown("### 🎭 通过 5 次选择，生成独一无二的故事")
    # 显示当前使用的背景
    custom_bg = st.text_area(
        "故事背景：",
        value=st.session_state.custom_background,
        height=160,
        placeholder="描述你想要的未来世界背景...",
        key="background_editor",
        disabled=not _is_debug_enabled(),
    )

    if _is_debug_enabled():
        st.caption("调试模式：可直接修改背景文本；或点击下方按钮随机生成（侧边栏可选择模型）。")
    else:
        st.caption("请点击下方按钮随机生成背景。")
    if st.button("🎲 随机生成背景", use_container_width=True):
        combo = random_background_text()
        # 合并到背景顶部，便于修改
        st.session_state.custom_background = combo
        write_log("background_randomized", {"combo": combo}, st.session_state.session_id)
        st.rerun()

    # 保存修改后的背景
    if _is_debug_enabled() and custom_bg != st.session_state.custom_background:
        st.session_state.custom_background = custom_bg
        st.success("✅ 故事背景已更新！")
    

    
    # 调试信息：显示当前状态
    if _is_debug_enabled():
        st.write("🔍 调试信息：")
        st.write(f"- 角色: {st.session_state.get('dynamic_role', 'None')}")
        st.write(f"- 选择数量: {len(st.session_state.get('dynamic_picks', []))}")
        st.write(f"- 问题数量: {len(st.session_state.get('dynamic_questions', []))}")
        st.write(f"- 提前结束: {st.session_state.get('dynamic_early_end', False)}")
    
    if st.session_state.get("dynamic_early_end"):
        st.info("🏁 当前状态：已选择提前结束")
    
    # 第0层：动态生成角色
    if not st.session_state.dynamic_role:
        st.markdown("---")
        st.subheader("第 0 层：生成角色")
        
        if not st.session_state.dynamic_roles_list:
            if st.button("🎲 生成角色选项", use_container_width=True):
                with st.spinner("正在生成角色..."):
                    try:
                        background_text = st.session_state.custom_background
                        messages = build_dynamic_roles_messages(background_text, beijing_date_str())
                        response = generate_chat(messages, st.session_state.question_model, temperature=0.8)
                        roles = [role.strip() for role in response.strip().split('\n') if role.strip()]
                        if len(roles) >= 4:
                            st.session_state.dynamic_roles_list = roles[:4]
                            write_log("roles_generated", {"roles": st.session_state.dynamic_roles_list}, st.session_state.session_id)
                        else:
                            st.error("角色生成失败，请重试")
                    except Exception as e:
                        st.error(f"生成失败：{e}")
                        write_log("roles_generate_error", {"error": str(e)}, st.session_state.session_id)
                st.rerun()
        else:
            st.write("请选择你的身份：")
            role = st.radio(
                "角色选择：",
                options=st.session_state.dynamic_roles_list,
                horizontal=True,
                key="dynamic_role_radio",
            )
            if st.button("进入世界之门 →", use_container_width=True):
                st.session_state.dynamic_role = role
                write_log("role_chosen", {"role": role}, st.session_state.session_id)
                st.rerun()
        return

    # 中间状态：显示路径
    picks = st.session_state.dynamic_picks
    depth = len(picks)
    if depth > 0:
        st.markdown("---")
        
        # 显示之前的抉择
        if st.session_state.dynamic_questions:
            st.markdown("**之前的抉择：**")
            for i, (q, t_opt, f_opt) in enumerate(st.session_state.dynamic_questions):
                if i < len(picks):  # 确保索引安全
                    choice = picks[i]
                    st.write(f"第{i+1}层：{q}")
                    st.write(f"选择：**{t_opt if choice else f_opt}**")
                else:
                    st.write(f"第{i+1}层：{q} (尚未选择)")

    # 继续生成问题（最多 3 层）
    if depth < 3:
        st.markdown("---")
        st.subheader(f"第 {depth + 1} 层")
        
        # 如果还没有当前层的问题，自动生成一个（除非已选择提前结束）
        if depth >= len(st.session_state.dynamic_questions) and not st.session_state.dynamic_early_end:
            with st.spinner("正在生成问题..."):
                try:
                    # 构建之前的抉择历史
                    previous_choices = []
                    for i, (q, t_opt, f_opt) in enumerate(st.session_state.dynamic_questions):
                        if i < len(picks):  # 确保索引安全
                            choice = picks[i]
                            previous_choices.append((q, t_opt if choice else f_opt))
                        else:
                            # 如果还没有选择，跳过这个问题
                            continue

                    background_text = st.session_state.custom_background
                    messages = build_dynamic_question_messages(
                        background_text,
                        st.session_state.dynamic_role,
                        depth + 1,
                        previous_choices,
                        beijing_date_str()
                    )
                    response = generate_chat(messages, st.session_state.question_model, temperature=0.8)
                    question, true_opt, false_opt = parse_dynamic_response(response)

                    if question and true_opt and false_opt:
                        st.session_state.dynamic_questions.append((question, true_opt, false_opt))
                        write_log("question_generated", {"depth": depth + 1, "q": question, "T": true_opt, "F": false_opt, "model": st.session_state.question_model}, st.session_state.session_id)
                        st.rerun()
                    else:
                        st.error("问题生成失败，请点击重试。")
                except Exception as e:
                    st.error(f"生成失败：{e}")
                    write_log("question_generate_error", {"depth": depth + 1, "error": str(e)}, st.session_state.session_id)
            if st.button("重试生成问题", use_container_width=True, key=f"retry_gen_{depth}"):
                write_log("question_retry", {"depth": depth + 1}, st.session_state.session_id)
                st.rerun()
            return
            # 注意：这里不return，让代码继续执行到完成状态判断
        
        # 显示当前层的问题
        if depth < len(st.session_state.dynamic_questions):
            q, true_opt, false_opt = st.session_state.dynamic_questions[depth]
            st.write(q)
            
            choice = st.radio(
                "选择：",
                options=[True, False],
                index=0,
                format_func=lambda x: true_opt if x else false_opt,
                horizontal=True,
                key=f"dynamic_choice_{depth}",
            )
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button("下一步 →", use_container_width=True):
                    st.session_state.dynamic_picks.append(bool(choice))
                    st.session_state.dynamic_novel_text = None
                    write_log("choice_next", {"depth": depth + 1, "choice": bool(choice)}, st.session_state.session_id)
                    st.rerun()
            with col3:
                if st.button("← 上一步", use_container_width=True, disabled=(len(picks) == 0)):
                    if picks:
                        picks.pop()
                    st.session_state.dynamic_novel_text = None
                    write_log("choice_back", {"depth": depth, "picks_len": len(picks)}, st.session_state.session_id)
                    st.rerun()
            with col4:
                if st.button("🏁 结束选择", use_container_width=True, type="secondary", key=f"end_early_choice_{depth}"):
                    # 标记为提前结束
                    st.session_state.dynamic_early_end = True
                    if _is_debug_enabled():
                        st.write("正在结束选择...")  # 调试信息
                    write_log("early_end", {"depth": depth, "path": "".join("T" if x else "F" for x in picks)}, st.session_state.session_id)
                    st.rerun()
            with col2:
                if st.button("换一个问题", use_container_width=True):
                    # 重新生成当前层的问题（不清空历史选择）
                    with st.spinner("正在重新生成本题..."):
                        try:
                            # 已完成的抉择历史（到上一题为止）
                            previous_choices = []
                            for i, (pq, pt, pf) in enumerate(st.session_state.dynamic_questions[:depth]):
                                if i < len(picks):
                                    choice = picks[i]
                                    previous_choices.append((pq, pt if choice else pf))

                            background_text = st.session_state.custom_background
                            messages = build_dynamic_question_messages(
                                background_text,
                                st.session_state.dynamic_role,
                                depth + 1,
                                previous_choices,
                                beijing_date_str()
                            )
                            response = generate_chat(messages, st.session_state.question_model, temperature=0.8)
                            question, true_opt, false_opt = parse_dynamic_response(response)
                            if question and true_opt and false_opt:
                                # 替换当前层的问题
                                st.session_state.dynamic_questions[depth] = (question, true_opt, false_opt)
                                write_log("question_regenerated", {"depth": depth + 1, "q": question, "T": true_opt, "F": false_opt, "model": st.session_state.question_model}, st.session_state.session_id)
                                st.rerun()
                            else:
                                st.error("重生成失败，请重试。")
                        except Exception as e:
                            st.error(f"重生成失败：{e}")
                            write_log("question_regenerate_error", {"depth": depth + 1, "error": str(e)}, st.session_state.session_id)
            # 注意：这里不return，让代码继续执行到完成状态判断
        # 如果当前层有问题但还没有选择，等待用户选择
        # 注意：这里不return，让代码继续执行到完成状态判断

    # 完成状态：走满3层或提前结束
    if depth == 3 or st.session_state.dynamic_early_end:  # 完成3个选择或提前结束
        st.markdown("---")
        if depth == 3:
            st.subheader("🎉 已完成所有决策")
        else:
            st.subheader("🏁 提前结束选择")

        # 计算路径字符串（用于日志与显示）
        path_str = "".join("T" if x else "F" for x in picks)

        # 调试信息
        if _is_debug_enabled():
            st.write(f"深度: {depth}, 提前结束: {st.session_state.dynamic_early_end}")
            st.write(f"条件判断: depth == 3 ({depth == 3}) OR early_end ({st.session_state.dynamic_early_end}) = {depth == 3 or st.session_state.dynamic_early_end}")
            st.markdown(f"**最终路径：** `{path_str}`")

            # 显示所有抉择
            st.markdown("**你的抉择历程：**")
            for i, (q, t_opt, f_opt) in enumerate(st.session_state.dynamic_questions):
                if i < len(picks):  # 确保索引安全
                    choice = picks[i]
                    st.write(f"第{i+1}层：{q}")
                    st.write(f"选择：**{t_opt if choice else f_opt}**")
                else:
                    st.write(f"第{i+1}层：{q} (尚未选择)")

        # 大纲与小说生成
        st.markdown("### 创作故事")
        st.caption("先有故事大纲，再据此创作中文短篇小说。")

        # 使用侧边栏设置
        style = st.session_state.dynamic_novel_style
        model = st.session_state.gen_model

        # 步骤1：生成大纲
        outline_generated_now = False  # 避免同一轮渲染重复展示
        if st.button("📝 生成故事大纲", use_container_width=True):
            try:
                decisions = []
                for i, (q, t_opt, f_opt) in enumerate(st.session_state.dynamic_questions):
                    if i < len(picks):
                        choice = picks[i]
                        decisions.append((q, t_opt if choice else f_opt))
                background_text = st.session_state.custom_background
                outline_messages = build_outline_messages(
                    background=background_text,
                    role=st.session_state.dynamic_role or "一位路人",
                    decisions=decisions,
                    date_cn=beijing_date_str(),
                )
                write_log(
                    "outline_prompt",
                    {"model": model, "path": path_str, "messages": outline_messages},
                    st.session_state.session_id,
                )
                placeholder = st.empty()
                acc = []
                for chunk in generate_chat_stream(outline_messages, model=model, temperature=0.6):
                    acc.append(chunk)
                    placeholder.write("".join(acc))
                outline_text = "".join(acc)
                st.session_state.dynamic_outline_text = outline_text
                write_log(
                    "outline_response",
                    {"model": model, "path": path_str, "text": outline_text},
                    st.session_state.session_id,
                )
                write_log("outline_generated", {"model": model, "path": path_str, "len": len(outline_text or "")}, st.session_state.session_id)
                outline_generated_now = True
            except Exception as e:
                st.error(f"大纲生成失败：{e}")
                write_log("outline_generate_error", {"model": model, "error": str(e)}, st.session_state.session_id)

        # 展示大纲
        if st.session_state.dynamic_outline_text and not outline_generated_now:
            st.markdown("#### 故事大纲（中英）")
            st.write(st.session_state.dynamic_outline_text)

        # 步骤2：基于大纲生成短篇
        generated_now = False
        if st.button("✨ 基于大纲生成短篇", use_container_width=True, disabled=not st.session_state.dynamic_outline_text):
            try:
                if not st.session_state.dynamic_outline_text:
                    st.warning("请先生成大纲")
                else:
                    novel_messages = build_novel_messages(
                        outline_text=st.session_state.dynamic_outline_text,
                        style=style,
                    )
                    write_log(
                        "novel_prompt",
                        {"model": model, "style": style, "path": path_str, "messages": novel_messages},
                        st.session_state.session_id,
                    )
                    placeholder = st.empty()
                    acc = []
                    for chunk in generate_chat_stream(novel_messages, model=model, temperature=0.9 if style == "诙谐幽默" else 0.8):
                        acc.append(chunk)
                        placeholder.write("".join(acc))
                    text = "".join(acc)
                    st.session_state.dynamic_novel_text = text
                    write_log(
                        "novel_response",
                        {"model": model, "style": style, "path": path_str, "text": text},
                        st.session_state.session_id,
                    )
                    write_log("novel_generated", {"model": model, "style": style, "path": path_str, "len": len(text or "")}, st.session_state.session_id)
                    generated_now = True
            except Exception as e:
                st.error(f"短篇生成失败：{e}")
                write_log("novel_generate_error", {"model": model, "error": str(e)}, st.session_state.session_id)

        if st.session_state.dynamic_novel_text and not generated_now:
            st.markdown("#### 生成结果")
            st.write(st.session_state.dynamic_novel_text)

        # 下载/回退/重置
        col_dl, col_rerun = st.columns(2)
        with col_dl:
            # 仅当已生成短篇时提供下载
            md_text = None
            if st.session_state.dynamic_novel_text:
                role_name = st.session_state.dynamic_role or "主人公"
                bg = st.session_state.custom_background
                story_text = st.session_state.dynamic_novel_text
                md_text = (
                    f"# 短篇故事\n\n"
                    f"- 主人公：{role_name}\n"
                    f"- 路径：{path_str}\n"
                    f"- 模型：{model}\n"
                    f"- 日期：{beijing_date_str()}\n\n"
                    f"## 背景\n\n{bg}\n\n---\n\n{story_text}\n"
                )
            st.download_button(
                "⬇️ 下载故事",
                data=(md_text or "").encode("utf-8"),
                file_name="story.md",
                mime="text/markdown",
                disabled=not bool(st.session_state.dynamic_novel_text),
                use_container_width=True,
            )
        with col_rerun:
            if st.button("重新开始", use_container_width=True, key="dynamic_reset"):
                st.session_state.dynamic_picks = []
                st.session_state.dynamic_questions = []
                st.session_state.dynamic_novel_text = None
                st.session_state.dynamic_outline_text = None
                st.session_state.dynamic_role = None
                st.session_state.dynamic_early_end = False
                st.session_state.custom_background = STORY_BACKGROUND  # 重置为默认背景
                st.rerun()