# -*- coding: utf-8 -*-
    """
    [L3] 未来之门 — Streamlit 1.47.0 版
    -----------------------------------
    运行：
        pip install -r requirements.txt
        streamlit run app.py

    功能：
    - A/B（或更多）分支式文字游戏
    - URL ?node=xxx 可分享当前进度
    - 终局导出旅程 JSON，支持重开
    - “AI 辅助”节点：若设置 OPENAI_API_KEY 且安装 openai，则在线生成两条建议；否则使用内置离线模板
    配置：
    - 故事结构在 story.yaml，可直接编辑内容增删节点
    - 主题在 .streamlit/config.toml
    """
    from __future__ import annotations

    import json, hashlib, random
    from pathlib import Path
    from datetime import datetime
    from typing import Dict, Any, List

    import streamlit as st
    import yaml

    st.set_page_config(page_title="[L3]未来之门", page_icon="🌀", layout="centered")

    # ------------------------------
    # 数据加载
    # ------------------------------
    @st.cache_data(show_spinner=False)
    def load_story(path: str = "story.yaml") -> Dict[str, Any]:
        p = Path(path)
        if not p.exists():
            return {}
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("story", {})

    STORY = load_story()

    # ------------------------------
    # AI 辅助建议（本地/在线）
    # ------------------------------
    def _seed_from_path(path: List[str]) -> int:
        s = "/".join(path)
        import hashlib
        h = hashlib.sha1(s.encode("utf-8")).hexdigest()
        return int(h[:8], 16)

    def ai_suggest_local(context: str, seed: int) -> List[str]:
        random.seed(seed)
        verbs_a = ["点燃", "启动", "唤醒", "联结", "加速"]
        verbs_b = ["校准", "分层", "缓释", "隔离", "回溯"]
        goals = ["知识共享网络", "自治协作体", "城市边缘算力", "低功耗传感网", "异构数据湖"]
        risks = ["数据偏倚", "供应链波动", "能耗攀升", "治理失效", "模型漂移"]
        a = f"【乐观】优先{random.choice(verbs_a)}核心能力，聚焦 {random.choice(goals)}，以开源社区为引擎抢占窗口期。"
        b = f"【谨慎】先{random.choice(verbs_b)}关键风险，针对 {random.choice(risks)} 设红线与闸门，小步快跑验证假设。"
        return [a, b]

    def ai_suggest_online(context: str) -> List[str] | None:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY")
            if not api_key:
                return None
            try:
                from openai import OpenAI
            except Exception:
                return None
            client = OpenAI(api_key=api_key)
            prompt = f"请基于如下背景，给出两条中文建议：一条偏乐观推进，一条偏谨慎防守，每条不超过60字。背景：{context}"
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "你是项目顾问。输出两条要点，前缀用【乐观】与【谨慎】。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
            )
            text = resp.choices[0].message.content or ""
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if len(lines) < 2:
                parts = text.replace("。", "。
").splitlines()
                lines = [p.strip() for p in parts if p.strip()][:2]
            return lines[:2] if lines else None
        except Exception:
            return None

    # ------------------------------
    # 状态初始化 / 跳转
    # ------------------------------
    def init_state(story: Dict[str, Any]):
        url_node = st.query_params.get("node", None)
        if "node" not in st.session_state:
            st.session_state.node = url_node if (url_node and url_node in story) else "start"
        if "path" not in st.session_state:
            st.session_state.path = [st.session_state.node]

    def goto(node_id: str):
        st.session_state.node = node_id
        if not st.session_state.path or st.session_state.path[-1] != node_id:
            st.session_state.path.append(node_id)
        st.query_params["node"] = node_id
        st.rerun()

    # ------------------------------
    # 渲染单个节点
    # ------------------------------
    def render_node(node_id: str, story: Dict[str, Any]):
        node = story.get(node_id)
        if not node:
            st.error(f"未找到节点：{node_id}")
            return

        with st.container(border=True):
            st.markdown(node.get("text", ""))

        # 可选：AI 辅助建议
        if node.get("ai_assist"):
            ctx = " / ".join(st.session_state.path)
            with st.status("AI 辅助生成中", expanded=False) as s:
                suggestions = ai_suggest_online(ctx)
                if not suggestions:
                    suggestions = ai_suggest_local(ctx, _seed_from_path(st.session_state.path))
                for sug in suggestions:
                    st.write("• " + sug)
                s.update(label="AI 辅助建议已生成", state="complete")

        # 终局
        if node.get("end"):
            c1, c2 = st.columns(2)
            with c1:
                if st.button("再来一次", use_container_width=True):
                    st.session_state.node = "start"
                    st.session_state.path = ["start"]
                    st.query_params["node"] = "start"
                    st.rerun()
            with c2:
                data = json.dumps(
                    {"path": st.session_state.path, "finished_at": datetime.now().isoformat()},
                    ensure_ascii=False, indent=2
                )
                st.download_button(
                    "下载我的旅程路径",
                    data=data,
                    file_name="l3_path.json",
                    mime="application/json",
                    use_container_width=True,
                )
            return

        # 选项
        opts = node.get("options", [])
        if not opts:
            st.warning("该节点没有配置选项。")
            return
        cols = st.columns(len(opts))
        for i, opt in enumerate(opts):
            label = opt.get("label", f"选项 {i+1}")
            target = opt.get("next")
            with cols[i]:
                if st.button(label, key=f"choice_{node_id}_{i}", use_container_width=True):
                    if target and target in story:
                        goto(target)
                    else:
                        st.error(f"无效的跳转目标：{target}")

    # ------------------------------
    # 页面结构
    # ------------------------------
    if not STORY:
        st.error("未找到或无法解析 story.yaml。请检查项目根目录是否存在 story.yaml 文件。")
    else:
        st.title("[L3]未来之门")
        st.caption("Made with Streamlit 1.47.0 · 文字游戏 · A/B 分支 · 可分享链接")

        with st.expander("作者/调试工具", expanded=False):
            st.write("当前 URL 结点：", st.query_params.get("node"))
            st.write("节点总数：", len(STORY))
            if st.button("重置进度"):
                st.session_state.clear()
                st.query_params.clear()
                st.rerun()

        init_state(STORY)
        render_node(st.session_state.node, STORY)