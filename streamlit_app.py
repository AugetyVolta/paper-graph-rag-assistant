import json
import re
import tempfile
import html
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

from app import (
    INDEX_FILE,
    SearchEngine,
    build_graph,
    call_llm,
    clip,
    compact_papers,
    deepseek_chat,
    get_deepseek_config,
    load_env_file,
    local_answer,
)


st.set_page_config(
    page_title="顶会论文检索系统",
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_env_file()

st.markdown(
    """
    <style>
    .block-container { padding-top: 0.8rem; padding-bottom: 0.8rem; }
    div[data-testid="stMetric"] {
        border: 1px solid #e3e8ef;
        border-radius: 8px;
        padding: 10px 12px;
        background: #ffffff;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 8px;
    }
    .aix-muted { color: #647084; font-size: 0.9rem; }
    .aix-title { font-size: 1.05rem; font-weight: 700; line-height: 1.35; }
    .aix-toolbar {
        border: 1px solid #e3e8ef;
        border-radius: 8px;
        padding: 12px 14px 6px 14px;
        background: #fbfcfd;
        margin-bottom: 12px;
    }
    .aix-page-title {
        font-size: 1.35rem;
        font-weight: 750;
        line-height: 1.25;
        margin: 0 0 0.15rem 0;
        color: #182033;
    }
    .aix-page-caption {
        color: #697386;
        font-size: 0.92rem;
        margin: 0 0 0.75rem 0;
    }
    .aix-view-header {
        margin: 0 0 0.85rem 0;
        padding: 0.35rem 0 0.12rem 0;
        overflow: visible;
    }
    .aix-view-title {
        color: #182033;
        font-size: 1.72rem;
        font-weight: 700;
        line-height: 1.45;
        letter-spacing: 0;
        margin: 0;
        padding: 0.05rem 0;
        overflow: visible;
        white-space: normal;
    }
    .aix-view-caption {
        color: #697386;
        font-size: 0.94rem;
        line-height: 1.45;
        margin: 0;
    }
    section[data-testid="stSidebar"] h1 {
        font-size: 1.35rem;
        line-height: 1.2;
    }
    section[data-testid="stSidebar"] {
        min-width: 285px !important;
    }
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 0.75rem !important;
    }
    .aix-sidebar-brand {
        padding: 0 0 0.55rem 0;
        margin-bottom: 0.25rem;
        border-bottom: 1px solid #dbe3ee;
    }
    .aix-sidebar-brand-title {
        color: #182033;
        font-size: 1.18rem;
        font-weight: 760;
        line-height: 1.2;
        margin: 0;
        white-space: nowrap;
    }
    .aix-sidebar-brand-subtitle {
        color: #697386;
        font-size: 0.76rem;
        line-height: 1.25;
        margin-top: 0.18rem;
    }
    .st-key-qa_chat_shell {
        position: relative;
        overflow: hidden !important;
    }
    .st-key-qa_chat_input_panel {
        background: #ffffff;
        margin-top: 0;
        padding-top: 0.6rem;
        border-top: 1px solid #e6eaf0;
        box-sizing: border-box;
        width: 100%;
        max-width: 100%;
    }
    .st-key-qa_chat_input_panel * {
        max-width: 100%;
        box-sizing: border-box;
    }
    .st-key-qa_chat_input_panel div[data-testid="stVerticalBlock"] {
        gap: 0.36rem;
    }
    .st-key-qa_chat_input_panel textarea {
        min-height: 60px !important;
    }
    .st-key-library_search_form button[kind="primaryFormSubmit"] {
        min-height: 38px;
        margin-top: 0 !important;
    }
    .aix-form-action-label {
        color: #313847;
        font-size: 0.875rem;
        line-height: 1.35;
        margin: 0 0 0.42rem 0;
        padding-top: 0.05rem;
    }
    .aix-mindmap {
        border: 1px solid #dce3ee;
        border-radius: 8px;
        background: #fbfcfe;
        padding: 16px;
        margin-top: 8px;
    }
    .aix-mindmap-center {
        border: 1px solid #c9d7ea;
        background: #eef5ff;
        border-radius: 8px;
        padding: 14px 16px;
        font-weight: 760;
        color: #182033;
        line-height: 1.35;
        text-align: center;
        margin-bottom: 14px;
    }
    .aix-mindmap-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
    }
    .aix-mindmap-card {
        border: 1px solid #e2e8f0;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        background: #ffffff;
        padding: 12px 12px 10px 12px;
        min-height: 150px;
    }
    .aix-mindmap-card:nth-child(2) { border-left-color: #10b981; }
    .aix-mindmap-card:nth-child(3) { border-left-color: #f59e0b; }
    .aix-mindmap-card:nth-child(4) { border-left-color: #ef4444; }
    .aix-mindmap-card:nth-child(5) { border-left-color: #8b5cf6; }
    .aix-mindmap-card:nth-child(6) { border-left-color: #06b6d4; }
    .aix-mindmap-card-title {
        font-weight: 730;
        color: #182033;
        margin-bottom: 8px;
    }
    .aix-mindmap-card ul {
        margin: 0;
        padding-left: 1.05rem;
    }
    .aix-mindmap-card li {
        color: #334155;
        font-size: 0.9rem;
        line-height: 1.45;
        margin-bottom: 5px;
    }
    @media (max-width: 1100px) {
        .aix-mindmap-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 720px) {
        .aix-mindmap-grid { grid-template-columns: 1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="正在加载 ICLR 2026 论文索引...")
def load_engine():
    return SearchEngine(INDEX_FILE)


engine = load_engine()


@st.cache_resource
def get_detail_note_executor():
    return ThreadPoolExecutor(max_workers=2)


DETAIL_NOTE_EXECUTOR = get_detail_note_executor()

DIRECTIONS = {
    "LLM Agent": ["agent", "agents", "tool use", "tools", "planning", "webarena", "workflow", "multi-agent"],
    "RAG": ["rag", "retrieval", "retrieval-augmented", "retrieve", "retriever", "dense retrieval"],
    "Knowledge Graph": ["knowledge graph", "graph rag", "graph retrieval", "entity", "relation", "kg"],
    "Reasoning": ["reasoning", "math", "chain-of-thought", "cot", "inference", "proof", "gsm8k", "mmlu"],
    "Multimodal": ["multimodal", "vision-language", "vlm", "image", "video", "audio", "visual"],
    "Tool Use": ["tool", "function calling", "api", "environment", "execution"],
    "Evaluation": ["benchmark", "evaluation", "dataset", "metric", "leaderboard"],
    "Safety": ["safety", "alignment", "jailbreak", "privacy", "fairness", "robustness"],
    "Reinforcement Learning": ["reinforcement learning", "rl", "ppo", "policy", "reward"],
    "Generative Models": ["diffusion", "flow matching", "generation", "generative", "image generation"],
}

METHOD_PATTERNS = {
    "RAG": ["rag", "retrieval"],
    "GraphRAG": ["graphrag", "graph rag", "graph retrieval", "knowledge graph"],
    "Agent Planning": ["planning", "planner", "agent"],
    "Tool Use": ["tool", "api", "function calling"],
    "Chain-of-Thought": ["chain-of-thought", "cot", "reasoning"],
    "RLHF/RL": ["rlhf", "ppo", "reinforcement learning", "reward model"],
    "Diffusion": ["diffusion", "flow matching"],
    "Multimodal Fusion": ["multimodal", "vision-language", "vlm"],
}

DATASET_PATTERNS = [
    "HotpotQA",
    "GSM8K",
    "MMLU",
    "HumanEval",
    "MBPP",
    "WebArena",
    "MiniGrid",
    "BabyAI",
    "CIFAR",
    "ImageNet",
    "COCO",
    "VQA",
    "MathVista",
    "GPQA",
    "ARC",
    "HellaSwag",
]


def ensure_state():
    # Remove keys from older UI versions that used widget-backed state names.
    for stale_key in ["graph_custom_topic"]:
        st.session_state.pop(stale_key, None)

    defaults = {
        "messages": [],
        "last_results": [],
        "last_result_ids": [],
        "compare_ids": [],
        "selected_paper_id": None,
        "detail_notes": {},
        "detail_note_jobs": {},
        "mind_maps": {},
        "mind_map_jobs": {},
        "inline_detail_mode": "summary",
        "paper_browser_query": "我想找和大语言模型、检索增强、知识图谱有关，适合做课程展示的论文",
        "current_page": "论文调研工作台",
        "suppress_auto_browser": False,
        "workspace_answer": "",
        "workspace_answer_sources": [],
        "last_search_plan": "",
        "last_search_query_used": "",
        "pending_workspace_search": None,
        "pending_detail_note": None,
        "qa_history": [],
        "assistant_question": "",
        "clear_assistant_question_on_next_run": False,
        "assistant_scope": "当前检索结果",
        "compare_ai_goal": "帮我挑几篇方法路线不同、适合课堂横向比较的论文",
        "clear_compare_goal_on_next_run": False,
        "compare_ai_note": "",
        "compare_ai_job": None,
        "graph_results": [],
        "graph_result_ids": [],
        "graph_total_count": 0,
        "graph_active_topic": "",
        "graph_topic_choice": "",
        "graph_custom_input": "",
        "library_query": "",
        "library_page_size": 24,
        "library_page": 1,
        "library_results": [],
        "library_detail_dialog_id": None,
        "trend_query": "",
        "trend_note": "",
        "trend_signature": "",
        "pending_trend_summary": None,
        "discovery_example_choice": "选择一个示例需求",
        "assistant_sessions": [
            {
                "id": "chat-1",
                "title": "默认对话",
                "messages": [],
                "sources": [],
            }
        ],
        "active_chat_id": "chat-1",
        "chat_counter": 1,
        "graph_query": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


ensure_state()


def paper_blob(paper):
    review = paper.get("reviews", {})
    return " ".join(
        [
            paper.get("title", ""),
            paper.get("primary_area", ""),
            " ".join(paper.get("keywords", [])),
            paper.get("abstract", ""),
            review.get("meta_review", ""),
            " ".join(review.get("summaries", [])),
            " ".join(review.get("strengths", [])),
            " ".join(review.get("weaknesses", [])),
            " ".join(review.get("questions", [])),
            " ".join(review.get("comments", [])),
        ]
    ).lower()


def classify_paper(paper):
    text = paper_blob(paper)
    scores = {}
    for direction, terms in DIRECTIONS.items():
        scores[direction] = sum(1 for term in terms if term in text)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else (paper.get("primary_area") or "Other")


def classify_results(results):
    groups = defaultdict(list)
    for paper in results:
        groups[classify_paper(paper)].append(paper)
    return dict(sorted(groups.items(), key=lambda item: len(item[1]), reverse=True))


def detect_methods(paper):
    text = paper_blob(paper)
    methods = []
    for method, patterns in METHOD_PATTERNS.items():
        if any(pattern in text for pattern in patterns):
            methods.append(method)
    return methods[:4]


def detect_datasets(paper):
    text = paper.get("abstract", "") + " " + " ".join(paper.get("keywords", [])) + " " + paper.get("title", "")
    found = []
    for dataset in DATASET_PATTERNS:
        if re.search(rf"\b{re.escape(dataset)}\b", text, flags=re.IGNORECASE):
            found.append(dataset)
    return found[:4]


def pdf_url(paper):
    pdf = paper.get("pdf", "")
    if pdf.startswith("http"):
        return pdf
    if pdf.startswith("/"):
        return "https://openreview.net" + pdf
    return ""


def short_authors(paper, limit=5):
    authors = paper.get("authors", [])
    if len(authors) > limit:
        return ", ".join(authors[:limit]) + " 等"
    return ", ".join(authors)


def add_compare_id(paper_id):
    if paper_id and paper_id not in st.session_state.compare_ids:
        st.session_state.compare_ids.append(paper_id)


def remove_compare_id(paper_id):
    st.session_state.compare_ids = [pid for pid in st.session_state.compare_ids if pid != paper_id]


def render_paper_card(paper, key_prefix="paper", show_actions=True):
    review = paper.get("reviews", {})
    with st.container(border=True):
        title = paper.get("title", "Untitled")
        st.markdown(f"**{title}**")
        st.caption(short_authors(paper))
        cols = st.columns([1, 1, 1])
        cols[0].metric("相似度", paper.get("score", "-"))
        cols[1].metric("评分", review.get("avg_rating", "暂无"))
        cols[2].metric("评审数", review.get("review_count", 0))
        st.caption(f"方向：{paper.get('primary_area') or '未知'}")
        if paper.get("keywords"):
            st.markdown(" ".join([f"`{kw}`" for kw in paper.get("keywords", [])[:6]]))
        st.write(clip(paper.get("abstract", ""), 260))
        links = []
        if paper.get("openreview_url"):
            links.append(f"[OpenReview]({paper['openreview_url']})")
        if pdf_url(paper):
            links.append(f"[PDF]({pdf_url(paper)})")
        if links:
            st.markdown(" | ".join(links))
        if show_actions:
            c1, c2 = st.columns(2)
            if c1.button("查看详情", key=f"{key_prefix}_detail_{paper['id']}"):
                st.session_state.selected_paper_id = paper["id"]
                st.session_state.inline_detail_mode = "summary"
                st.rerun()
            if c2.button("加入对比", key=f"{key_prefix}_compare_{paper['id']}"):
                add_compare_id(paper["id"])
                st.toast("已加入对比列表")


def render_compact_result(paper, key_prefix="result"):
    review = paper.get("reviews", {})
    is_selected = paper.get("id") == st.session_state.selected_paper_id
    border_label = "当前选中" if is_selected else ""
    with st.container(border=True):
        st.markdown(f"<div class='aix-title'>{paper.get('title')}</div>", unsafe_allow_html=True)
        st.caption(short_authors(paper, limit=4))
        meta = [
            classify_paper(paper),
            f"评分 {review.get('avg_rating', '暂无')}",
            f"相似度 {paper.get('score', '-')}",
        ]
        if border_label:
            meta.insert(0, border_label)
        st.markdown(" · ".join([f"`{item}`" for item in meta]))
        st.write(clip(paper.get("abstract", ""), 150))
        c1, c2, c3 = st.columns([1, 1, 1])
        if c1.button("选择", key=f"{key_prefix}_select_{paper['id']}", use_container_width=True):
            st.session_state.selected_paper_id = paper["id"]
            st.session_state.inline_detail_mode = "summary"
            st.rerun()
        if c2.button("对比", key=f"{key_prefix}_compare_{paper['id']}", use_container_width=True):
            add_compare_id(paper["id"])
            st.toast("已加入对比列表")
        c3.link_button("原文", paper.get("openreview_url") or "https://openreview.net", use_container_width=True)


def paper_dataframe(results):
    rows = []
    for paper in results:
        review = paper.get("reviews", {})
        rows.append(
            {
                "标题": paper.get("title"),
                "自动分类": classify_paper(paper),
                "研究方向": paper.get("primary_area"),
                "关键词": ", ".join(paper.get("keywords", [])[:5]),
                "评分": review.get("avg_rating"),
                "相似度": paper.get("score"),
                "OpenReview": paper.get("openreview_url"),
            }
        )
    return pd.DataFrame(rows)


def run_rag_answer(question, results, history=None):
    instruction = (
        "你是一个顶会论文调研助手。请只根据给定 ICLR 2026 论文材料回答，不要编造不存在的论文。"
        "回答时必须做到：1. 先直接回答用户问题；2. 按研究方向分类；"
        "3. 每个观点尽量对应具体论文标题；4. 给出代表论文；5. 总结趋势、不足和可继续探索的问题。"
        "不要用 [1]、[2] 这样的序号引用。如果证据不足，明确说明证据不足。"
    )
    return call_llm(question, results, history=history, instruction=instruction) or local_answer(question, results, engine.stats)


def llm_from_papers(task, papers, temperature=0.2):
    messages = [
        {
            "role": "system",
            "content": (
                "你是 ICLR 2026 论文调研助手。只能依据给定论文 JSON 回答，不能编造。"
                "请用中文输出结构化内容，并用论文标题或短标题说明依据，不要用 [1]、[2] 这样的序号引用。"
            ),
        },
        {
            "role": "user",
            "content": f"任务：{task}\n\n论文 JSON：{json.dumps(compact_papers(papers, limit=10), ensure_ascii=False)}",
        },
    ]
    return deepseek_chat(messages, temperature=temperature) or "DeepSeek 暂时没有返回结果，请稍后重试。"


def render_sources(results, title="来源论文"):
    with st.expander(title, expanded=True):
        if not results:
            st.info("暂无来源论文。")
            return
        for paper in results:
            render_paper_card(paper, key_prefix=f"src_{paper['id']}")


def matches_extra_filters(paper, extra_filters):
    venue_choice = extra_filters.get("venue")
    if venue_choice and venue_choice != "全部类型":
        venue = paper.get("venue", "").lower()
        status = paper.get("status", "").lower()
        if venue_choice == "Oral" and "oral" not in venue:
            return False
        if venue_choice == "Spotlight" and "spotlight" not in venue:
            return False
        if venue_choice == "Poster" and "poster" not in venue:
            return False
        if venue_choice == "Submitted / Other" and any(x in venue for x in ["oral", "spotlight", "poster"]):
            return False
        if venue_choice == "Accepted" and "accepted" not in status:
            return False

    keyword = extra_filters.get("keyword", "").strip().lower()
    if keyword:
        blob = paper_blob(paper)
        terms = [term for term in re.split(r"[\s,，;；]+", keyword) if term]
        if terms and not all(term in blob for term in terms):
            return False
    return True


def search_papers(query, top_k, filters=None, extra_filters=None, candidate_ids=None):
    extra_filters = extra_filters or {}
    raw_k = max(top_k * 4, 30)
    results = engine.search(query, top_k=raw_k, filters=filters or {}, candidate_ids=candidate_ids)
    filtered = [paper for paper in results if matches_extra_filters(paper, extra_filters)]
    if not filtered and extra_filters:
        return results[:top_k]
    return filtered[:top_k]


def browse_papers(query, limit=None, filters=None, extra_filters=None):
    query = (query or "").strip()
    if query:
        return search_papers(query, top_k=limit or 500, filters=filters, extra_filters=extra_filters)

    rows = []
    for paper in engine.papers:
        if filters and filters.get("area") and paper.get("primary_area") != filters["area"]:
            continue
        result = dict(paper)
        result.pop("search_text", None)
        result["score"] = "-"
        if matches_extra_filters(result, extra_filters or {}):
            rows.append(result)
        if limit and len(rows) >= limit:
            break
    return rows


DISCOVERY_EXAMPLES = [
    "我想找几篇适合初学者讲清楚的大语言模型论文",
    "我想了解 RAG 和知识图谱结合的论文",
    "我想找和 AI Agent、工具调用、任务规划有关的论文",
    "我想找多模态大模型方向，实验比较充分的论文",
    "我想了解 reviewer 常批评哪些问题，比如实验不足或泛化不足",
    "我想找安全、对齐、隐私相关的论文",
]


WORKSPACE_HEIGHT = 920
RESULT_LIST_HEIGHT = 700
CHAT_HISTORY_HEIGHT = 300


def render_page_header(title, caption):
    caption_html = f'<p class="aix-view-caption">{caption}</p>' if caption else ""
    st.markdown(
        f"""
        <div class="aix-view-header">
            <div class="aix-view-title" role="heading" aria-level="1">{title}</div>
            {caption_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_plain_page_header(title, caption):
    st.write("")
    st.markdown(f"## {title}")
    if caption:
        st.caption(caption)


def inject_panel_resizer(specs):
    payload = json.dumps(specs)
    components.html(
        f"""
        <script>
        (() => {{
            const specs = {payload};
            const parentWindow = window.parent;
            const parentDocument = parentWindow.document;

            function clearPanel(el) {{
                el.style.removeProperty("height");
                el.style.removeProperty("min-height");
                el.style.removeProperty("max-height");
            }}

            function resizePanels() {{
                const viewportHeight = parentWindow.innerHeight || parentDocument.documentElement.clientHeight;
                const viewportWidth = parentWindow.innerWidth || parentDocument.documentElement.clientWidth;
                for (const spec of specs) {{
                    const el = parentDocument.querySelector(spec.selector);
                    if (!el) continue;
                    if (viewportWidth <= (spec.mobileBreakpoint || 900)) {{
                        clearPanel(el);
                        continue;
                    }}
                    const rect = el.getBoundingClientRect();
                    const height = Math.max(spec.minHeight || 360, viewportHeight - rect.top - (spec.bottomGap || 18));
                    const value = Math.round(height) + "px";
                    el.style.setProperty("height", value, "important");
                    el.style.setProperty("min-height", value, "important");
                    el.style.setProperty("max-height", value, "important");
                }}
            }}

            resizePanels();
            parentWindow.addEventListener("resize", resizePanels);
            const timer = parentWindow.setInterval(resizePanels, 700);
            parentWindow.setTimeout(() => parentWindow.clearInterval(timer), 12000);
        }})();
        </script>
        """,
        height=0,
    )


def inject_chat_dock_layout():
    components.html(
        """
        <script>
        (() => {
            const parentWindow = window.parent;
            const parentDocument = parentWindow.document;

            function setImportant(el, prop, value) {
                el.style.setProperty(prop, value, "important");
            }

            function clearDock(shell, history, input) {
                for (const el of [shell, history, input]) {
                    if (!el) continue;
                    for (const prop of ["position", "left", "right", "bottom", "z-index", "height", "min-height", "max-height", "overflow", "background", "padding-left", "padding-right", "width", "max-width", "box-sizing"]) {
                        el.style.removeProperty(prop);
                    }
                }
            }

            function layoutChat() {
                const shell = parentDocument.querySelector(".st-key-qa_chat_shell");
                const history = parentDocument.querySelector(".st-key-qa_chat_history");
                const input = parentDocument.querySelector(".st-key-qa_chat_input_panel");
                if (!shell || !history || !input) return;

                const viewportWidth = parentWindow.innerWidth || parentDocument.documentElement.clientWidth;
                if (viewportWidth <= 900) {
                    clearDock(shell, history, input);
                    return;
                }

                setImportant(shell, "position", "relative");
                setImportant(shell, "overflow", "hidden");
                setImportant(input, "position", "absolute");
                setImportant(input, "left", "16px");
                setImportant(input, "right", "16px");
                setImportant(input, "bottom", "16px");
                setImportant(input, "z-index", "5");
                setImportant(input, "width", "calc(100% - 32px)");
                setImportant(input, "max-width", "calc(100% - 32px)");
                setImportant(input, "box-sizing", "border-box");
                setImportant(input, "background", "#ffffff");
                setImportant(input, "padding-left", "0");
                setImportant(input, "padding-right", "0");

                const shellRect = shell.getBoundingClientRect();
                const historyRect = history.getBoundingClientRect();
                const inputRect = input.getBoundingClientRect();
                const available = shellRect.bottom - historyRect.top - inputRect.height - 34;
                const height = Math.max(140, available);
                const value = Math.round(height) + "px";
                setImportant(history, "height", value);
                setImportant(history, "min-height", value);
                setImportant(history, "max-height", value);
                setImportant(history, "overflow", "auto");
            }

            layoutChat();
            parentWindow.addEventListener("resize", layoutChat);
            const timer = parentWindow.setInterval(layoutChat, 500);
            parentWindow.setTimeout(() => parentWindow.clearInterval(timer), 12000);
        })();
        </script>
        """,
        height=0,
    )


def render_example_buttons(examples, state_key, prefix):
    cols = st.columns(3)
    for idx, example in enumerate(examples):
        if cols[idx % 3].button(example, key=f"{prefix}_{idx}", use_container_width=True):
            st.session_state[state_key] = example
            st.rerun()


def parse_json_object(text):
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if match:
        cleaned = match.group(0)
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def ai_plan_search_query(user_query):
    query = (user_query or "").strip() or "large language model"
    messages = [
        {
            "role": "system",
            "content": (
                "你是 ICLR 2026 论文检索意图规划器。你的任务不是回答论文问题，"
                "而是把用户中文自然语言需求改写成适合本地论文索引检索的英文关键词组合。"
                "只能输出 JSON，不要输出 Markdown。JSON 字段："
                "query: 英文关键词组合，覆盖论文标题、摘要、关键词、评审意见中可能出现的术语；"
                "focus: 用中文简述你理解的检索目标；"
                "tags: 3 到 6 个方向标签；"
                "reason: 用一句中文说明为什么这样检索。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"用户需求：{query}\n"
                f"可用方向标签参考：{', '.join(DIRECTIONS.keys())}\n"
                "请优先加入常见英文术语缩写，例如 LLM, RAG, agent, retrieval, knowledge graph, multimodal, reasoning, safety。"
            ),
        },
    ]
    content = deepseek_chat(messages, temperature=0.1, timeout=45)
    data = parse_json_object(content)
    if not data:
        return {
            "query": query,
            "focus": "DeepSeek 暂时没有返回可解析的检索规划，已直接使用原始需求检索。",
            "tags": [],
            "reason": "",
        }

    planned_query = str(data.get("query") or query).strip()
    return {
        "query": planned_query or query,
        "focus": str(data.get("focus") or query).strip(),
        "tags": [str(tag).strip() for tag in data.get("tags", []) if str(tag).strip()][:6],
        "reason": str(data.get("reason") or "").strip(),
    }


def run_workspace_search(user_query, top_k, filters, extra_filters, use_ai=True):
    if use_ai:
        plan = ai_plan_search_query(user_query)
        query_used = plan["query"]
    else:
        plan = {
            "query": user_query or "large language model",
            "focus": "直接按输入内容检索。",
            "tags": [],
            "reason": "",
        }
        query_used = plan["query"]

    results = search_papers(query_used, top_k=top_k, filters=filters, extra_filters=extra_filters)
    st.session_state.last_results = results
    st.session_state.last_result_ids = [paper["id"] for paper in results]
    st.session_state.selected_paper_id = results[0]["id"] if results else None
    st.session_state.last_search_plan = plan
    st.session_state.last_search_query_used = query_used
    st.session_state.suppress_auto_browser = False
    return results


def use_discovery_example():
    choice = st.session_state.get("discovery_example_choice", "")
    if choice and choice != "选择一个示例需求":
        st.session_state.paper_browser_query = choice


def clear_workspace_search():
    st.session_state.last_results = []
    st.session_state.last_result_ids = []
    st.session_state.selected_paper_id = None
    st.session_state.workspace_answer = ""
    st.session_state.workspace_answer_sources = []
    st.session_state.last_search_plan = ""
    st.session_state.last_search_query_used = ""
    st.session_state.paper_browser_query = ""
    st.session_state.suppress_auto_browser = True


def handle_pending_workspace_search():
    pending_search = st.session_state.get("pending_workspace_search")
    if not pending_search:
        return

    with st.spinner("DeepSeek 正在规划检索并更新结果..."):
        run_workspace_search(
            pending_search.get("query"),
            pending_search.get("top_k", 8),
            pending_search.get("filters") or {},
            pending_search.get("extra_filters") or {},
            use_ai=True,
        )
        st.session_state.pending_workspace_search = None
    st.rerun()


def generate_detail_note(paper):
    return llm_from_papers(
        "请解读这篇论文：包括一句话总结、研究问题、核心方法、主要贡献、实验说明和可能局限。不要评价它是否适合课程汇报。",
        [paper],
    )


def normalize_mind_map_data(data, paper):
    if not isinstance(data, dict):
        raise ValueError("DeepSeek 没有返回合法的思维导图 JSON。")

    center = str(data.get("center") or paper.get("title") or "Untitled").strip()
    raw_branches = data.get("branches", [])
    branches = []
    if isinstance(raw_branches, list):
        for branch in raw_branches:
            if not isinstance(branch, dict):
                continue
            title = str(branch.get("title") or "").strip()
            raw_items = branch.get("items", [])
            if isinstance(raw_items, str):
                raw_items = [raw_items]
            items = [clip(str(item).strip(), 120) for item in raw_items if str(item).strip()]
            if title and items:
                branches.append({"title": title, "items": items[:4]})

    if len(branches) < 4:
        raise ValueError("DeepSeek 返回的思维导图结构不完整。")
    return {"center": center, "branches": branches[:6]}


def generate_mind_map_data(paper):
    messages = [
        {
            "role": "system",
            "content": (
                "你是论文阅读流程设计助手。请只依据给定 ICLR 2026 论文 JSON 生成中文思维导图数据。"
                "只能输出 JSON，不要输出 Markdown。JSON 格式："
                '{"center":"论文短标题","branches":[{"title":"研究问题","items":["..."]},'
                '{"title":"核心思路","items":["..."]},{"title":"方法流程","items":["..."]},'
                '{"title":"实验评估","items":["..."]},{"title":"主要贡献","items":["..."]},'
                '{"title":"局限与追问","items":["..."]}]}。'
                "每个分支 2 到 4 条，语言短、清楚、适合课堂展示。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(compact_papers([paper], limit=1), ensure_ascii=False),
        },
    ]
    content = deepseek_chat(messages, temperature=0.15, timeout=90)
    data = parse_json_object(content)
    return normalize_mind_map_data(data, paper)


def start_mind_map_job(map_key, paper):
    jobs = st.session_state.mind_map_jobs
    if map_key not in jobs:
        jobs[map_key] = DETAIL_NOTE_EXECUTOR.submit(generate_mind_map_data, paper)


def start_detail_note_job(note_key, paper):
    jobs = st.session_state.detail_note_jobs
    if note_key not in jobs:
        jobs[note_key] = DETAIL_NOTE_EXECUTOR.submit(generate_detail_note, paper)
    st.session_state.pending_detail_note = note_key


@st.fragment(run_every="2s")
def render_pending_detail_note(note_key):
    job = st.session_state.detail_note_jobs.get(note_key)
    if not job:
        return

    if job.done():
        try:
            st.session_state.detail_notes[note_key] = job.result()
        except Exception as exc:
            st.session_state.detail_notes[note_key] = f"AI 解读生成失败：{exc}"
        st.session_state.detail_note_jobs.pop(note_key, None)
        if st.session_state.pending_detail_note == note_key:
            st.session_state.pending_detail_note = None
        st.rerun()

    with st.container(border=True):
        st.markdown("**AI 解读生成中**")
        st.progress(35)


def clean_markmap_text(value):
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = text.replace("<", "‹").replace(">", "›")
    return text.replace("\n", " ")


def mind_map_markdown(data):
    lines = [f"# {clean_markmap_text(data.get('center', '论文思维导图'))}"]
    for branch in data.get("branches", [])[:6]:
        title = clean_markmap_text(branch.get("title", ""))
        if not title:
            continue
        lines.append(f"## {title}")
        for item in branch.get("items", [])[:4]:
            item_text = clean_markmap_text(item)
            if item_text:
                lines.append(f"- {item_text}")
    return "\n".join(lines)


def markmap_html(data):
    markdown = mind_map_markdown(data).replace("</script", "<\\/script")
    return f"""
    <!doctype html>
    <html>
    <head>
    <meta charset="utf-8" />
    <style>
    body {{
        margin: 0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: transparent;
    }}
    .markmap {{
        position: relative;
        width: 100%;
        height: 540px;
        border: 1px solid #dce3ee;
        border-radius: 8px;
        background: #fbfcfe;
        overflow: hidden;
    }}
    .markmap > svg {{
        width: 100%;
        height: 100%;
        display: block;
    }}
    </style>
    </head>
    <body>
        <div class="markmap">
            <script type="text/template">
{markdown}
            </script>
        </div>
        <script src="https://cdn.jsdelivr.net/npm/markmap-autoloader@latest"></script>
    </body>
    </html>
    """


def render_mind_map(data):
    if data.get("error"):
        st.error(data["error"])
        return
    components.html(markmap_html(data), height=580, scrolling=True)


@st.fragment(run_every="2s")
def render_pending_mind_map(map_key, paper):
    job = st.session_state.mind_map_jobs.get(map_key)
    if not job:
        return

    if job.done():
        try:
            st.session_state.mind_maps[map_key] = job.result()
        except Exception as exc:
            st.session_state.mind_maps[map_key] = {"error": f"思维导图生成失败：{exc}"}
        st.session_state.mind_map_jobs.pop(map_key, None)
        st.rerun()

    with st.container(border=True):
        st.markdown("**思维导图生成中**")
        st.progress(40)


def get_active_session():
    sessions = st.session_state.assistant_sessions
    if not sessions:
        st.session_state.assistant_sessions = [
            {"id": "chat-1", "title": "默认对话", "messages": [], "sources": []}
        ]
        st.session_state.active_chat_id = "chat-1"
        st.session_state.chat_counter = 1
        return st.session_state.assistant_sessions[0]

    for session in sessions:
        if session["id"] == st.session_state.active_chat_id:
            return session

    st.session_state.active_chat_id = sessions[-1]["id"]
    return sessions[-1]


def create_chat_session():
    st.session_state.chat_counter += 1
    session_id = f"chat-{st.session_state.chat_counter}"
    session = {
        "id": session_id,
        "title": f"新对话 {st.session_state.chat_counter}",
        "messages": [],
        "sources": [],
    }
    st.session_state.assistant_sessions.append(session)
    st.session_state.active_chat_id = session_id
    st.session_state.assistant_question = ""
    st.session_state.workspace_answer = ""
    st.session_state.workspace_answer_sources = []


def delete_active_session():
    active_id = st.session_state.active_chat_id
    st.session_state.assistant_sessions = [
        session for session in st.session_state.assistant_sessions if session["id"] != active_id
    ]
    if not st.session_state.assistant_sessions:
        st.session_state.chat_counter += 1
        st.session_state.assistant_sessions = [
            {
                "id": f"chat-{st.session_state.chat_counter}",
                "title": "新对话",
                "messages": [],
                "sources": [],
            }
        ]
    st.session_state.active_chat_id = st.session_state.assistant_sessions[-1]["id"]


def clear_assistant_input():
    st.session_state.assistant_question = ""


def clear_widget_before_render(flag_key, widget_key):
    if st.session_state.get(flag_key):
        st.session_state[widget_key] = ""
        st.session_state[flag_key] = False


def session_label(session):
    count = sum(1 for message in session.get("messages", []) if message.get("role") == "user")
    return f"{session['title']} ({count})"


def resolve_answer_sources(question, scope, filters, extra_filters, top_k, follow_scope):
    selected_paper = engine.get_paper(st.session_state.selected_paper_id) if st.session_state.selected_paper_id else None
    if scope == "选中论文" and selected_paper:
        return [selected_paper]

    if scope == "当前检索结果" and st.session_state.last_result_ids:
        sources = search_papers(
            question,
            top_k=top_k,
            filters=filters,
            extra_filters=extra_filters,
            candidate_ids=st.session_state.last_result_ids if follow_scope else None,
        )
        return sources or st.session_state.last_results[:top_k]

    return search_papers(question, top_k=top_k, filters=filters, extra_filters=extra_filters)


def render_chat_panel(filters, extra_filters, top_k, follow_scope, history_height=CHAT_HISTORY_HEIGHT, history_key=None):
    session = get_active_session()
    st.subheader("AI 助手")

    sessions_by_id = {item["id"]: item for item in st.session_state.assistant_sessions}
    ids = list(sessions_by_id.keys())
    if st.session_state.active_chat_id not in ids:
        st.session_state.active_chat_id = ids[-1]
    active_index = ids.index(st.session_state.active_chat_id) if st.session_state.active_chat_id in ids else 0
    selected_id = st.selectbox(
        "历史会话",
        ids,
        index=active_index,
        format_func=lambda session_id: session_label(sessions_by_id[session_id]),
    )
    if selected_id != st.session_state.active_chat_id:
        st.session_state.active_chat_id = selected_id
        st.rerun()

    h1, h2 = st.columns(2)
    h1.button("新对话", key="assistant_new_session", use_container_width=True, on_click=create_chat_session)
    h2.button("删除当前", key="assistant_delete_session", use_container_width=True, on_click=delete_active_session)

    session = get_active_session()
    source_papers = [engine.get_paper(paper_id) for paper_id in session.get("sources", [])]
    source_papers = [paper for paper in source_papers if paper]
    display_history_height = history_height
    history_container_kwargs = {"height": display_history_height, "border": False}
    if history_key:
        history_container_kwargs["key"] = history_key
    with st.container(**history_container_kwargs):
        if not session["messages"]:
            st.empty()
        else:
            for message in session["messages"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            if source_papers:
                with st.expander("本轮依据论文", expanded=False):
                    for paper in source_papers[:6]:
                        st.markdown(f"**{paper['title']}**")
                        if paper.get("openreview_url"):
                            st.markdown(f"[OpenReview]({paper['openreview_url']})")

    with st.container(border=False, key="qa_chat_input_panel"):
        scope = st.radio(
            "回答范围",
            ["当前检索结果", "选中论文", "全库重新检索"],
            key="assistant_scope",
            horizontal=False,
        )
        clear_widget_before_render("clear_assistant_question_on_next_run", "assistant_question")
        question = st.text_area(
            "问题",
            key="assistant_question",
            height=64,
            placeholder="例如：这些论文主要分成哪几类？哪几篇最容易理解？",
        )
        c1, c2 = st.columns(2)
        ask_clicked = c1.button("发送", key="assistant_send", use_container_width=True, disabled=not question)
        c2.button("清空输入", key="assistant_clear_input", use_container_width=True, on_click=clear_assistant_input)

    if ask_clicked:
        with st.status("DeepSeek 正在阅读证据...", expanded=False):
            sources = resolve_answer_sources(question, scope, filters, extra_filters, top_k, follow_scope)
            history = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in session["messages"][-6:]
            ]
            answer = run_rag_answer(question, sources, history=history)
            session["messages"].append({"role": "user", "content": question})
            session["messages"].append({"role": "assistant", "content": answer})
            session["sources"] = [paper["id"] for paper in sources]
            if session["title"].startswith("新对话") or session["title"] == "默认对话":
                session["title"] = clip(question, 18)
            st.session_state.workspace_answer_sources = sources
            st.session_state.workspace_answer = answer
            st.session_state.clear_assistant_question_on_next_run = True
        st.rerun()


def sidebar_filters():
    stats = engine.stats
    st.sidebar.title("控制台")
    m1, m2 = st.sidebar.columns(2)
    m1.metric("论文", stats.get("paper_count", 0))
    m2.metric("评审", stats.get("papers_with_reviews", 0))
    api_key, _, model = get_deepseek_config()
    st.sidebar.caption(f"LLM：{'DeepSeek ' + model if api_key else '未配置'}")

    top_areas = ["全部方向"] + [area for area, _ in stats.get("top_areas", [])]
    with st.sidebar.expander("检索筛选", expanded=True):
        area = st.selectbox("方向筛选（可不选）", top_areas)
        venue_choice = st.selectbox("接收类型（可不选）", ["全部类型", "Poster", "Oral", "Spotlight", "Accepted", "Submitted / Other"])
        keyword = st.text_input("额外关键词（可不填）", placeholder="例如 retrieval / safety / image")
        top_k = st.slider("检索论文数量", 3, 100, 8, step=1)
        follow_scope = st.checkbox("追问优先使用上一轮论文", value=True)

    filters = {}
    if area != "全部方向":
        filters["area"] = area
    extra_filters = {"venue": venue_choice, "keyword": keyword}
    return filters, extra_filters, top_k, follow_scope


def render_selected_paper_detail(paper, key_prefix="inline"):
    review = paper.get("reviews", {})
    st.subheader("论文详情")
    with st.container(border=True):
        st.markdown(f"### {paper.get('title')}")
        st.caption(short_authors(paper, limit=10))
        m1, m2, m3 = st.columns(3)
        m1.metric("分类", classify_paper(paper))
        m2.metric("评分", review.get("avg_rating", "暂无"))
        m3.metric("评审", review.get("review_count", 0))
        if paper.get("keywords"):
            st.markdown(" ".join([f"`{kw}`" for kw in paper.get("keywords", [])[:8]]))
        c1, c2, c3 = st.columns(3)
        if c1.button("加入对比", key=f"{key_prefix}_add_compare_{paper['id']}", use_container_width=True):
            add_compare_id(paper["id"])
            st.toast("已加入对比列表")
        c2.link_button("OpenReview", paper.get("openreview_url") or "https://openreview.net", use_container_width=True)
        if pdf_url(paper):
            c3.link_button("PDF", pdf_url(paper), use_container_width=True)

    tabs = st.tabs(["摘要", "思维导图", "评审"])
    with tabs[0]:
        st.write(paper.get("abstract", ""))
        note_key = paper["id"]
        if st.button("生成 AI 解读", key=f"{key_prefix}_note_{paper['id']}", use_container_width=True):
            start_detail_note_job(note_key, paper)
            st.toast("已开始生成 AI 解读")
            st.rerun()
        if note_key in st.session_state.detail_notes:
            st.markdown(st.session_state.detail_notes[note_key])
        elif note_key in st.session_state.detail_note_jobs:
            render_pending_detail_note(note_key)

    with tabs[1]:
        map_key = paper["id"]
        if st.button("生成论文思维导图", key=f"{key_prefix}_mindmap_{paper['id']}", use_container_width=True):
            start_mind_map_job(map_key, paper)
            st.toast("已开始生成思维导图")
            st.rerun()
        if map_key in st.session_state.mind_maps:
            render_mind_map(st.session_state.mind_maps[map_key])
        elif map_key in st.session_state.mind_map_jobs:
            render_pending_mind_map(map_key, paper)
        else:
            st.empty()

    with tabs[2]:
        st.markdown("**Reviewer 肯定点**")
        for item in review.get("strengths", [])[:3]:
            st.write("- " + item)
        st.markdown("**Reviewer 质疑点**")
        for item in review.get("weaknesses", [])[:3]:
            st.write("- " + item)
        if review.get("meta_review"):
            with st.expander("Meta Review"):
                st.write(review.get("meta_review"))


def close_library_detail_dialog():
    st.session_state.library_detail_dialog_id = None


@st.dialog("论文详情", width="large", on_dismiss=close_library_detail_dialog)
def render_library_detail_dialog(paper_id):
    paper = engine.get_paper(paper_id)
    if not paper:
        st.warning("没有找到这篇论文，可能索引已更新。")
        if st.button("关闭", key="library_dialog_missing_close"):
            close_library_detail_dialog()
            st.rerun()
        return

    render_selected_paper_detail(paper, key_prefix=f"library_dialog_{paper_id}")
    c1, _ = st.columns([1, 3])
    if c1.button("关闭", key=f"library_dialog_close_{paper_id}", use_container_width=True):
        close_library_detail_dialog()
        st.rerun()


def page_qa():
    filters, extra_filters, top_k, follow_scope = sidebar_filters()
    st.markdown(
        """
        <style>
        :root {
            --aix-qa-lower-pane-height: clamp(560px, calc(100vh - 410px), 1200px);
            --aix-qa-chat-pane-height: clamp(720px, calc(100vh - 178px), 1280px);
            --aix-qa-chat-history-height: 300px;
        }
        .st-key-qa_results_pane,
        .st-key-qa_results_pane > div[data-testid="stVerticalBlockBorderWrapper"],
        .st-key-qa_detail_pane,
        .st-key-qa_detail_pane > div[data-testid="stVerticalBlockBorderWrapper"] {
            height: var(--aix-qa-lower-pane-height) !important;
            min-height: var(--aix-qa-lower-pane-height) !important;
            max-height: var(--aix-qa-lower-pane-height) !important;
            overflow-y: auto !important;
            scrollbar-gutter: stable;
        }
        .st-key-qa_chat_shell,
        .st-key-qa_chat_shell > div[data-testid="stVerticalBlockBorderWrapper"] {
            height: var(--aix-qa-chat-pane-height) !important;
            min-height: var(--aix-qa-chat-pane-height) !important;
            max-height: var(--aix-qa-chat-pane-height) !important;
            overflow: hidden !important;
            scrollbar-gutter: stable;
        }
        .st-key-qa_chat_history,
        .st-key-qa_chat_history > div[data-testid="stVerticalBlockBorderWrapper"] {
            height: var(--aix-qa-chat-history-height) !important;
            min-height: var(--aix-qa-chat-history-height) !important;
            max-height: var(--aix-qa-chat-history-height) !important;
            overflow-y: auto !important;
            scrollbar-gutter: stable;
        }
        @media (max-width: 900px) {
            .st-key-qa_results_pane,
            .st-key-qa_detail_pane,
            .st-key-qa_chat_shell,
            .st-key-qa_chat_history {
                height: auto !important;
                min-height: 0 !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    render_plain_page_header(
        "论文调研工作台",
        "",
    )
    inject_panel_resizer(
        [
            {"selector": ".st-key-qa_results_pane", "minHeight": 560, "bottomGap": 18},
            {"selector": ".st-key-qa_detail_pane", "minHeight": 560, "bottomGap": 18},
            {"selector": ".st-key-qa_chat_shell", "minHeight": 720, "bottomGap": 18},
        ]
    )
    inject_chat_dock_layout()

    workspace_col, chat_col = st.columns([0.72, 0.28], gap="large")

    with workspace_col:
        with st.container(border=True, key="qa_search_panel"):
            st.markdown("**用自然语言描述你想找的论文**")
            top_cols = st.columns([0.72, 0.28], gap="medium")
            with top_cols[0]:
                search_query = st.text_area(
                    "自然语言需求",
                    key="paper_browser_query",
                    label_visibility="collapsed",
                    height=92,
                    placeholder="例如：我想找几篇容易讲清楚、实验比较充分的大语言模型论文",
                )
            with top_cols[1]:
                st.selectbox(
                    "示例需求",
                    ["选择一个示例需求"] + DISCOVERY_EXAMPLES,
                    key="discovery_example_choice",
                    label_visibility="collapsed",
                )
                st.button("使用示例", key="use_discovery_example", use_container_width=True, on_click=use_discovery_example)

            search_cols = st.columns([1, 1, 3.6], gap="medium")
            if search_cols[0].button("AI 检索", key="workspace_ai_search", use_container_width=True):
                st.session_state.pending_workspace_search = {
                    "query": search_query,
                    "top_k": top_k,
                    "filters": filters,
                    "extra_filters": extra_filters,
                }
                st.toast("已开始 AI 检索")
            search_cols[1].button("清空", key="clear_workspace_search", use_container_width=True, on_click=clear_workspace_search)
            handle_pending_workspace_search()

        if not st.session_state.last_results and not st.session_state.suppress_auto_browser:
            results = search_papers(st.session_state.paper_browser_query, top_k=top_k, filters=filters, extra_filters=extra_filters)
            st.session_state.last_results = results
            st.session_state.last_result_ids = [paper["id"] for paper in results]
            st.session_state.selected_paper_id = results[0]["id"] if results else None
            st.session_state.last_search_plan = {
                "focus": "默认按输入内容进行本地检索；点击 AI 检索后会先调用 DeepSeek 改写检索意图。",
                "query": st.session_state.paper_browser_query,
                "tags": [],
                "reason": "",
            }
            st.session_state.last_search_query_used = st.session_state.paper_browser_query

        result_col, detail_col = st.columns([0.45, 0.55], gap="large")
        with result_col:
            with st.container(height=RESULT_LIST_HEIGHT, border=True, key="qa_results_pane"):
                st.subheader("检索结果")
                if st.session_state.last_results:
                    groups = classify_results(st.session_state.last_results)
                    st.caption(" / ".join([f"{name} {len(papers)}" for name, papers in groups.items()]))
                    for paper in st.session_state.last_results:
                        render_compact_result(paper, key_prefix="workspace")
                else:
                    st.info("暂无检索结果。")

        with detail_col:
            with st.container(height=RESULT_LIST_HEIGHT, border=True, key="qa_detail_pane"):
                selected_paper = engine.get_paper(st.session_state.selected_paper_id) if st.session_state.selected_paper_id else None
                if selected_paper:
                    render_selected_paper_detail(selected_paper, key_prefix="workspace_detail")
                else:
                    st.subheader("论文详情")
                    st.info("未选择论文。")

    with chat_col:
        with st.container(height=WORKSPACE_HEIGHT, border=True, key="qa_chat_shell"):
            render_chat_panel(
                filters,
                extra_filters,
                top_k,
                follow_scope,
                history_height=CHAT_HISTORY_HEIGHT,
                history_key="qa_chat_history",
            )


def compare_terms(paper):
    terms = {classify_paper(paper)}
    terms.update(detect_methods(paper))
    terms.update(detect_datasets(paper))
    terms.update(str(keyword).lower() for keyword in paper.get("keywords", [])[:6])
    return {term for term in terms if term}


def diversity_score(paper, selected):
    if not selected:
        return 1.0
    current = compare_terms(paper)
    if not current:
        return 0.5
    max_overlap = 0.0
    for item in selected:
        other = compare_terms(item)
        union = current | other
        overlap = len(current & other) / len(union) if union else 0.0
        max_overlap = max(max_overlap, overlap)
    return 1 - max_overlap


def recommend_compare_papers(pool, limit=4):
    candidates = []
    seen = set()
    for paper in pool:
        paper_id = paper.get("id")
        if not paper_id or paper_id in seen:
            continue
        seen.add(paper_id)
        candidates.append(paper)

    selected = []
    target = min(limit, len(candidates))
    while len(selected) < target:
        best_paper = None
        best_score = -1
        for paper in candidates:
            if paper in selected:
                continue
            rating = numeric_rating(paper) or 0
            retrieval_score = float(paper.get("score") or 0)
            score = diversity_score(paper, selected) * 2.2 + rating * 0.16 + min(retrieval_score, 80) * 0.01
            if score > best_score:
                best_score = score
                best_paper = paper
        if not best_paper:
            break
        selected.append(best_paper)
    return selected


def generate_compare_note(goal, papers):
    return llm_from_papers(
        f"用户对比目标：{goal}\n请说明为什么这几篇论文适合横向比较，并给出建议比较维度。输出 4-6 条要点，不要太长。",
        papers,
    )


def start_compare_note_job(goal, papers):
    st.session_state.compare_ai_note = ""
    st.session_state.compare_ai_job = DETAIL_NOTE_EXECUTOR.submit(generate_compare_note, goal, papers)


@st.fragment(run_every="2s")
def render_pending_compare_note():
    job = st.session_state.get("compare_ai_job")
    if not job:
        return

    if job.done():
        try:
            st.session_state.compare_ai_note = job.result()
        except Exception as exc:
            st.session_state.compare_ai_note = f"AI 对比建议生成失败：{exc}"
        st.session_state.compare_ai_job = None
        st.rerun()

    with st.container(border=True):
        st.markdown("**AI 对比建议生成中**")
        st.progress(35)


def render_compare_ai_panel(filters, extra_filters, top_k):
    with st.container(border=True):
        st.markdown("**AI 辅助对比**")
        clear_widget_before_render("clear_compare_goal_on_next_run", "compare_ai_goal")
        goal = st.text_area(
            "对比目标",
            key="compare_ai_goal",
            height=88,
            placeholder="例如：帮我挑几篇 RAG、知识图谱、Agent 方法路线不同的论文做横向比较",
        )
        recommend_clicked = st.button("AI 推荐并加入对比", key="compare_ai_recommend", use_container_width=True)

    if recommend_clicked:
        query = (goal or st.session_state.paper_browser_query or "large language model").strip()
        pool = st.session_state.last_results or search_papers(
            query,
            top_k=max(top_k, 10),
            filters=filters,
            extra_filters=extra_filters,
        )
        if pool and not st.session_state.last_results:
            st.session_state.last_results = pool
            st.session_state.last_result_ids = [paper["id"] for paper in pool]
        recommended = recommend_compare_papers(pool, limit=4)
        if recommended:
            for paper in recommended:
                add_compare_id(paper["id"])
            start_compare_note_job(goal, recommended)
            st.session_state.clear_compare_goal_on_next_run = True
            st.toast("已推荐论文并开始生成对比建议")
            st.rerun()
        else:
            st.warning("可推荐论文不足。")

    if st.session_state.get("compare_ai_job"):
        render_pending_compare_note()
    elif st.session_state.compare_ai_note:
        with st.expander("AI 对比建议", expanded=True):
            st.markdown(st.session_state.compare_ai_note)


def page_compare():
    filters, extra_filters, top_k, _ = sidebar_filters()
    render_plain_page_header(
        "论文对比",
        "",
    )
    render_compare_ai_panel(filters, extra_filters, top_k)

    st.subheader("从上一轮检索结果添加")
    if st.session_state.last_results:
        titles = {paper["title"]: paper["id"] for paper in st.session_state.last_results}
        selected = st.multiselect("选择论文加入对比", list(titles.keys()))
        if st.button("加入所选论文"):
            for title in selected:
                add_compare_id(titles[title])
            st.toast("已加入对比列表")
    else:
        st.info("暂无上一轮检索结果。")

    compare_papers = [engine.get_paper(pid) for pid in st.session_state.compare_ids]
    compare_papers = [paper for paper in compare_papers if paper]

    if not compare_papers:
        st.warning("对比列表为空。")
        return

    st.subheader("当前对比列表")
    for paper in compare_papers:
        c1, c2 = st.columns([0.82, 0.18])
        c1.write(f"**{paper['title']}**")
        if c2.button("移除", key=f"remove_{paper['id']}"):
            remove_compare_id(paper["id"])
            st.rerun()

    base_rows = []
    for paper in compare_papers:
        review = paper.get("reviews", {})
        base_rows.append(
            {
                "论文": paper.get("title"),
                "自动分类": classify_paper(paper),
                "研究方向": paper.get("primary_area"),
                "方法线索": ", ".join(detect_methods(paper)) or "待分析",
                "数据集线索": ", ".join(detect_datasets(paper)) or "摘要中未明显出现",
                "评分": review.get("avg_rating"),
                "常见不足线索": clip(" ".join(review.get("weaknesses", [])[:1]), 120),
            }
        )
    st.dataframe(pd.DataFrame(base_rows), use_container_width=True, hide_index=True)

    if st.button("用 DeepSeek 生成完整对比表", disabled=len(compare_papers) < 2, use_container_width=True):
        with st.spinner("DeepSeek 正在做横向比较..."):
            st.markdown(
                llm_from_papers(
                    "请比较这些论文，并输出 Markdown 表格。列包括：研究问题、核心方法、创新点、实验设置/数据集、优点、不足、适合借鉴的地方。表格后总结这些论文代表的研究趋势。",
                    compare_papers[:5],
                )
            )


def graph_html(results):
    net = Network(height="650px", width="100%", directed=True, notebook=False, cdn_resources="in_line")
    net.barnes_hut(gravity=-2400, central_gravity=0.2, spring_length=120, spring_strength=0.03)

    def add_node(node_id, label, group, size=18):
        net.add_node(node_id, label=clip(label, 40), title=label, group=group, size=size)

    for paper in results:
        pid = f"paper:{paper['id']}"
        add_node(pid, paper["title"], "paper", size=24)
        direction = classify_paper(paper)
        did = f"direction:{direction}"
        add_node(did, direction, "direction", size=26)
        net.add_edge(pid, did, label="属于")

        for method in detect_methods(paper):
            mid = f"method:{method}"
            add_node(mid, method, "method", size=20)
            net.add_edge(pid, mid, label="使用")

        for dataset in detect_datasets(paper):
            dsid = f"dataset:{dataset}"
            add_node(dsid, dataset, "dataset", size=18)
            net.add_edge(pid, dsid, label="评估于")

        for keyword in paper.get("keywords", [])[:4]:
            kid = f"keyword:{keyword.lower()}"
            add_node(kid, keyword, "keyword", size=16)
            net.add_edge(pid, kid, label="关键词")

    similar_results = results[:20]
    for i, left in enumerate(similar_results):
        left_terms = set([kw.lower() for kw in left.get("keywords", [])])
        for right in similar_results[i + 1 :]:
            common = left_terms & set([kw.lower() for kw in right.get("keywords", [])])
            if len(common) >= 2:
                net.add_edge(f"paper:{left['id']}", f"paper:{right['id']}", label="相似")

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".html", encoding="utf-8") as fh:
        path = Path(fh.name)
    net.save_graph(str(path))
    return path.read_text(encoding="utf-8")


def numeric_rating(paper):
    value = paper.get("reviews", {}).get("avg_rating")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def graph_node_rows(results):
    rows = []
    for paper in results:
        paper_terms = [classify_paper(paper)]
        paper_terms.extend(detect_methods(paper))
        paper_terms.extend(detect_datasets(paper))
        paper_terms.extend(str(keyword).strip() for keyword in paper.get("keywords", [])[:5])
        for term in paper_terms:
            if not term:
                continue
            rows.append(
                {
                    "节点": term,
                    "论文": paper.get("title"),
                    "方向": classify_paper(paper),
                    "方法": ", ".join(detect_methods(paper)) or "摘要中未明显出现",
                    "关键词": ", ".join(paper.get("keywords", [])[:4]),
                    "摘要": clip(paper.get("abstract", ""), 220),
                }
            )
    return rows


def graph_overview(results):
    directions = Counter(classify_paper(paper) for paper in results)
    methods = Counter(method for paper in results for method in detect_methods(paper))
    datasets = Counter(dataset for paper in results for dataset in detect_datasets(paper))
    keywords = Counter(
        str(keyword).strip()
        for paper in results
        for keyword in paper.get("keywords", [])[:5]
        if str(keyword).strip()
    )
    return directions, methods, datasets, keywords


def set_graph_topic(label, query, results=None):
    st.session_state.graph_active_topic = label
    st.session_state.graph_query = query
    st.session_state.trend_query = f"请总结 ICLR 2026 中 {label} 方向的研究趋势"
    st.session_state.graph_selected_node = ""
    if results is not None:
        st.session_state.graph_results = results
        st.session_state.graph_result_ids = [paper["id"] for paper in results]


def run_graph_topic_search(label, query, top_k, filters, extra_filters):
    results = search_papers(query, top_k=max(top_k, 10), filters=filters, extra_filters=extra_filters)
    set_graph_topic(label, query, results)
    st.session_state.last_results = results
    st.session_state.last_result_ids = [paper["id"] for paper in results]


def use_last_results_for_graph():
    results = st.session_state.last_results[:12]
    label = "上一轮调研结果"
    query = st.session_state.last_search_query_used or st.session_state.paper_browser_query or label
    set_graph_topic(label, query, results)


def use_library_results_for_graph(results, label):
    selected = results[:12]
    if not selected:
        return
    query = st.session_state.library_query or label
    set_graph_topic(label, query, selected)
    st.session_state.last_results = selected
    st.session_state.last_result_ids = [paper["id"] for paper in selected]


def conference_area_options():
    return [area for area, _ in engine.stats.get("top_areas", []) if area]


def area_count_label(area):
    counts = dict(engine.stats.get("top_areas", []))
    count = counts.get(area)
    return f"{area} ({count})" if count else area


def conference_area_count(area):
    return dict(engine.stats.get("top_areas", [])).get(area, 0)


def apply_graph_topic_choice():
    st.session_state.graph_custom_input = ""


def generate_simple_graph(area, detail_text, top_k, filters, extra_filters):
    area = (area or "").strip()
    detail_text = (detail_text or "").strip()
    graph_filters = dict(filters or {})
    if area:
        graph_filters["area"] = area
    query = f"{area} {detail_text}".strip() or area or "machine learning"
    label = area if not detail_text else f"{area} / {detail_text}"
    results = search_papers(query, top_k=top_k, filters=graph_filters, extra_filters=extra_filters)
    set_graph_topic(label, query, results)
    st.session_state.graph_total_count = conference_area_count(area) if area else len(results)
    st.session_state.last_results = results
    st.session_state.last_result_ids = [paper["id"] for paper in results]
    st.session_state.trend_note = ""
    st.session_state.trend_signature = ""
    trend_query = (
        f"请基于 ICLR 2026 中 primary_area 为“{area or label}”的论文，总结“{label}”主题的研究趋势。"
        "同时解释图谱中论文、研究方向、方法、数据集、关键词之间反映出的关系。"
    )
    st.session_state.trend_query = trend_query
    st.session_state.pending_trend_summary = {
        "query": trend_query,
        "paper_ids": [paper.get("id") for paper in results[:10]],
        "signature": trend_query + "|" + "|".join(paper.get("id", "") for paper in results[:10]),
    }
    return results


def render_paper_library(filters, extra_filters):
    st.subheader("论文库浏览")

    with st.form("library_search_form"):
        c1, c2, c3 = st.columns([0.58, 0.20, 0.22], gap="medium")
        with c1:
            query = st.text_input(
                "搜索论文",
                key="library_query",
                placeholder="输入标题、关键词、作者或方向，例如 RAG / agent / safety",
            )
        with c2:
            page_size = st.selectbox(
                "每页显示",
                [12, 24, 36, 50],
                index=[12, 24, 36, 50].index(int(st.session_state.library_page_size)),
            )
            st.session_state.library_page_size = int(page_size)
        with c3:
            st.markdown("<div class='aix-form-action-label'>操作</div>", unsafe_allow_html=True)
            refresh_clicked = st.form_submit_button("搜索/刷新", use_container_width=True)

    if refresh_clicked or not st.session_state.library_results:
        search_limit = 1000 if query.strip() else None
        st.session_state.library_results = browse_papers(query, search_limit, filters=filters, extra_filters=extra_filters)
        st.session_state.library_page = 1

    results = st.session_state.library_results
    page_size = min(int(st.session_state.library_page_size), 50)
    total_pages = max(1, (len(results) + page_size - 1) // page_size)
    st.session_state.library_page = min(max(1, int(st.session_state.library_page)), total_pages)
    page = st.session_state.library_page
    start = (page - 1) * page_size
    page_results = results[start : start + page_size]
    st.caption(f"共 {len(results)} 篇匹配论文 · 当前第 {page} / {total_pages} 页")

    action_cols = st.columns([1, 1, 3])
    if action_cols[0].button("上一页", key="library_prev_page", disabled=page <= 1, use_container_width=True):
        st.session_state.library_page -= 1
        st.rerun()
    if action_cols[1].button("下一页", key="library_next_page", disabled=page >= total_pages, use_container_width=True):
        st.session_state.library_page += 1
        st.rerun()

    card_cols = st.columns(3)
    for idx, paper in enumerate(page_results):
        review = paper.get("reviews", {})
        with card_cols[idx % 3].container(border=True):
            st.markdown(f"**{clip(paper.get('title', 'Untitled'), 88)}**")
            st.caption(short_authors(paper, limit=4))
            meta = [
                classify_paper(paper),
                f"评分 {review.get('avg_rating', '暂无')}",
                f"相似度 {paper.get('score', '-')}",
            ]
            st.markdown(" · ".join([f"`{item}`" for item in meta]))
            if paper.get("keywords"):
                st.markdown(" ".join([f"`{kw}`" for kw in paper.get("keywords", [])[:4]]))
            st.write(clip(paper.get("abstract", ""), 170))
            b1, b2, b3 = st.columns(3)
            if b1.button("查看详情", key=f"library_detail_{paper['id']}", use_container_width=True):
                st.session_state.library_detail_dialog_id = paper["id"]
            if b2.button("加入对比", key=f"library_compare_{paper['id']}", use_container_width=True):
                add_compare_id(paper["id"])
                st.toast("已加入对比列表")
            b3.link_button("OpenReview", paper.get("openreview_url") or "https://openreview.net", use_container_width=True)

    if st.session_state.library_detail_dialog_id:
        render_library_detail_dialog(st.session_state.library_detail_dialog_id)


def render_graph_linked_browser(results):
    if not results:
        st.info("暂无图谱。")
        return

    directions, methods, datasets, keywords = graph_overview(results)
    st.subheader("图谱节点")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("论文节点", len(results))
    m2.metric("方向节点", len(directions))
    m3.metric("方法节点", len(methods))
    m4.metric("数据集节点", len(datasets))

    tabs = st.tabs(["方向", "方法", "数据集", "关键词"])
    counters = [directions, methods, datasets, keywords]
    prefixes = ["direction", "method", "dataset", "keyword"]
    for tab, counter, prefix in zip(tabs, counters, prefixes):
        with tab:
            if not counter:
                st.info("暂无节点。")
                continue
            cols = st.columns(3)
            for idx, (name, count) in enumerate(counter.most_common(12)):
                label = f"{name} ({count})"
                if cols[idx % 3].button(label, key=f"graph_node_{prefix}_{idx}_{name}", use_container_width=True):
                    st.session_state.graph_selected_node = name

    selected_value = st.session_state.get("graph_selected_node")
    if selected_value:
        rows = [
            row for row in graph_node_rows(results)
            if row["节点"].lower() == selected_value.lower()
        ]
        st.markdown(f"**关联论文：{selected_value}**")
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("暂无关联论文。")
    else:
        st.empty()


def render_trend_summary(query, results):
    st.subheader("研究趋势总结")
    trend_query = st.text_input(
        "趋势问题",
        key="trend_query",
        placeholder="例如：请总结 ICLR 2026 中 LLM Agent 方向的研究趋势",
    )
    signature = trend_query + "|" + "|".join(paper.get("id", "") for paper in results[:10])

    if st.button("用 DeepSeek 生成研究趋势总结", disabled=not results, key="trend_summary_button", use_container_width=True):
        st.session_state.pending_trend_summary = {
            "query": trend_query or query,
            "paper_ids": [paper.get("id") for paper in results[:10]],
            "signature": signature,
        }
        st.toast("已开始生成趋势总结")

    if st.session_state.trend_note and st.session_state.trend_signature == signature:
        with st.expander("AI 研究趋势总结", expanded=True):
            st.markdown(st.session_state.trend_note)
    elif st.session_state.trend_note:
        st.info("主题已变化。")


def handle_pending_trend_summary():
    pending = st.session_state.get("pending_trend_summary")
    if not pending:
        return

    papers = [engine.get_paper(paper_id) for paper_id in pending.get("paper_ids", [])]
    papers = [paper for paper in papers if paper]
    st.session_state.pending_trend_summary = None
    if not papers:
        return

    with st.status("DeepSeek 正在总结研究趋势...", expanded=True):
        st.session_state.trend_note = llm_from_papers(
            (
                f"用户问题：{pending.get('query')}\n"
                "请不要只列论文，要总结 3-5 个研究趋势。每个趋势包括：趋势标题、含义、代表论文、"
                "为什么适合课堂汇报。最后用一句话总结该方向从什么阶段走向什么阶段。"
            ),
            papers[:10],
        )
        st.session_state.trend_signature = pending.get("signature", "")
    st.rerun()


def render_simple_graph_workspace(filters, extra_filters, top_k):
    st.subheader("研究图谱")

    candidates = conference_area_options()
    if filters.get("area") in candidates:
        st.session_state.graph_topic_choice = filters["area"]
    if candidates and st.session_state.graph_topic_choice not in candidates:
        st.session_state.graph_topic_choice = candidates[0]

    with st.container(border=True):
        c1, c2 = st.columns([0.28, 0.72], gap="medium")
        with c1:
            if candidates:
                st.selectbox(
                    "会议方向",
                    candidates,
                    key="graph_topic_choice",
                    format_func=area_count_label,
                    on_change=apply_graph_topic_choice,
                )
            else:
                st.info("暂无会议方向。")
        with c2:
            detail_text = st.text_area(
                "细化主题（可选）",
                key="graph_custom_input",
                height=92,
                placeholder="例如：retrieval / graph learning / tool use / diffusion",
            )

        if st.button("生成知识图谱和研究趋势分析", key="graph_simple_generate", use_container_width=True, disabled=not candidates):
            with st.spinner("正在检索论文并构建知识图谱..."):
                generate_simple_graph(st.session_state.graph_topic_choice, detail_text, top_k, filters, extra_filters)
            st.toast("已生成知识图谱，并开始趋势分析")
            st.rerun()

    results = st.session_state.graph_results
    if not results:
        st.info("暂无图谱。")
        return

    directions, methods, datasets, _ = graph_overview(results)
    total_count = st.session_state.get("graph_total_count") or len(results)
    st.caption(
        f"当前主题：{st.session_state.graph_active_topic} · "
        f"方向总数 {total_count} 篇 · 当前筛选 {len(results)} 篇"
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("当前筛选", len(results))
    m2.metric("方向", len(directions))
    m3.metric("方法/数据集", len(methods) + len(datasets))

    st.subheader("知识图谱")
    components.html(graph_html(results), height=680, scrolling=True)

    with st.expander("图谱依据论文", expanded=False):
        rows = []
        for paper in results:
            review = paper.get("reviews", {})
            rows.append(
                {
                    "论文": paper.get("title"),
                    "自动分类": classify_paper(paper),
                    "方法线索": ", ".join(detect_methods(paper)) or "摘要中未明显出现",
                    "评分": review.get("avg_rating"),
                    "关键词": ", ".join(paper.get("keywords", [])[:4]),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("研究趋势分析")
    handle_pending_trend_summary()
    if st.session_state.trend_note:
        st.markdown(st.session_state.trend_note)
    else:
        st.empty()


def page_graph_and_review():
    filters, extra_filters, top_k, _ = sidebar_filters()
    render_plain_page_header(
        "论文库与研究图谱",
        "",
    )

    tab_library, tab_graph = st.tabs(["论文库浏览", "研究图谱"])

    with tab_library:
        render_paper_library(filters, extra_filters)

    with tab_graph:
        render_simple_graph_workspace(filters, extra_filters, top_k)


def main():
    pages = ["论文调研工作台", "论文对比", "论文库与研究图谱"]
    if st.session_state.current_page == "知识图谱与评审分析":
        st.session_state.current_page = "论文库与研究图谱"
    if st.session_state.current_page == "研究图谱与趋势总结":
        st.session_state.current_page = "论文库与研究图谱"
    if st.session_state.current_page not in pages:
        st.session_state.current_page = pages[0]
    st.sidebar.markdown(
        """
        <div class="aix-sidebar-brand">
            <div class="aix-sidebar-brand-title">顶会论文检索系统</div>
            <div class="aix-sidebar-brand-subtitle">ICLR 2026 · LLM + RAG</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio(
        "页面",
        pages,
        index=pages.index(st.session_state.current_page),
    )
    st.session_state.current_page = page

    if st.session_state.current_page == "论文调研工作台":
        page_qa()
    elif st.session_state.current_page == "论文对比":
        page_compare()
    else:
        page_graph_and_review()


if __name__ == "__main__":
    main()
