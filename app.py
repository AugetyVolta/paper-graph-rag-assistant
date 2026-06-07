import argparse
import json
import math
import mimetypes
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


INDEX_FILE = Path("data/iclr2026_index.json")
WEB_DIR = Path("web")
ENV_FILE = Path(".env")

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+\-]{1,}|[0-9]+(?:\.[0-9]+)?")
STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "are", "was", "were", "have", "has", "had",
    "into", "over", "under", "using", "use", "used", "can", "could", "would", "should", "about", "paper",
    "papers", "method", "model", "models", "learning", "based", "towards", "between", "through", "their",
    "they", "them", "than", "then", "also", "show", "shows", "study", "new", "via", "our",
}

CN_EXPANSIONS = {
    "大语言模型": "large language model llm foundation model",
    "语言模型": "language model llm",
    "多模态": "multimodal vision language",
    "知识图谱": "knowledge graph graph relation entity",
    "检索": "retrieval rag retrieve",
    "增强生成": "retrieval augmented generation rag",
    "智能体": "agent agents tool planning",
    "代理": "agent agents",
    "推理": "reasoning inference chain thought",
    "数学": "math mathematical reasoning",
    "代码": "code programming",
    "安全": "safety security robustness jailbreak",
    "对齐": "alignment preference reward",
    "强化学习": "reinforcement learning rl policy",
    "扩散": "diffusion generative",
    "图像": "image vision visual",
    "视频": "video",
    "语音": "speech audio",
    "机器人": "robot robotics embodied",
    "医学": "medical healthcare clinical",
    "蛋白": "protein biology",
    "推荐": "recommend recommendation",
    "综述": "survey overview",
    "缺点": "weakness limitation concern",
    "不足": "weakness limitation concern",
    "优点": "strength contribution",
    "评分": "rating score review",
    "评审": "review reviewer rating confidence",
    "接受": "accepted accept poster oral spotlight",
    "海报": "poster",
}


def load_env_file(path=ENV_FILE):
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
    if os.environ.get("AIX_DISABLE_PROXY", "").lower() in {"1", "true", "yes"}:
        for key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
            os.environ.pop(key, None)


def expand_query(query):
    additions = []
    lower = query.lower()
    for cn, en in CN_EXPANSIONS.items():
        if cn in query:
            additions.append(en)
    if "llm" in lower:
        additions.append("large language model foundation model")
    if "rag" in lower:
        additions.append("retrieval augmented generation")
    return f"{query} {' '.join(additions)}"


def tokenize(text):
    expanded = expand_query(text)
    tokens = []
    for token in TOKEN_RE.findall(expanded.lower()):
        token = token.strip("-_")
        if len(token) > 1 and token not in STOPWORDS:
            tokens.append(token)
    return tokens


def clip(text, limit):
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


class SearchEngine:
    def __init__(self, index_path):
        with index_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        self.metadata = payload.get("metadata", {})
        self.stats = payload.get("stats", {})
        self.papers = payload.get("papers", [])
        self.doc_tokens = []
        self.doc_lens = []
        self.df = Counter()
        self.avgdl = 1.0
        self._build()

    def _build(self):
        total_len = 0
        for paper in self.papers:
            text = " ".join([
                paper.get("title", ""),
                " ".join(paper.get("authors", [])),
                " ".join(paper.get("keywords", [])),
                paper.get("primary_area", ""),
                paper.get("abstract", ""),
                paper.get("reviews", {}).get("meta_review", ""),
                " ".join(paper.get("reviews", {}).get("summaries", [])),
                " ".join(paper.get("reviews", {}).get("strengths", [])),
                " ".join(paper.get("reviews", {}).get("weaknesses", [])),
                " ".join(paper.get("reviews", {}).get("questions", [])),
                " ".join(paper.get("reviews", {}).get("comments", [])),
            ])
            counts = Counter(tokenize(text))
            self.doc_tokens.append(counts)
            doc_len = sum(counts.values()) or 1
            self.doc_lens.append(doc_len)
            total_len += doc_len
            self.df.update(counts.keys())
        if self.papers:
            self.avgdl = total_len / len(self.papers)

    def search(self, query, top_k=8, filters=None, candidate_ids=None):
        filters = filters or {}
        candidate_set = set(candidate_ids) if candidate_ids else None
        q_tokens = tokenize(query)
        if not q_tokens:
            q_tokens = tokenize("large language model")
        q_counts = Counter(q_tokens)
        n_docs = max(len(self.papers), 1)
        k1 = 1.5
        b = 0.75
        scores = []
        query_lower = query.lower()

        for idx, paper in enumerate(self.papers):
            if candidate_set is not None and paper.get("id") not in candidate_set:
                continue
            if filters.get("area") and paper.get("primary_area") != filters["area"]:
                continue
            if filters.get("status") and paper.get("status") != filters["status"]:
                continue

            counts = self.doc_tokens[idx]
            score = 0.0
            for token, qf in q_counts.items():
                tf = counts.get(token, 0)
                if not tf:
                    continue
                df = self.df.get(token, 0)
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                denom = tf + k1 * (1 - b + b * self.doc_lens[idx] / self.avgdl)
                score += idf * (tf * (k1 + 1) / denom) * (1 + math.log(qf))

            title_lower = paper.get("title", "").lower()
            keywords_lower = " ".join(paper.get("keywords", [])).lower()
            area_lower = paper.get("primary_area", "").lower()
            for token in set(q_tokens):
                if token in title_lower:
                    score += 1.8
                if token in keywords_lower:
                    score += 1.2
                if token in area_lower:
                    score += 0.8
            if query_lower and query_lower in title_lower:
                score += 8.0

            if score > 0:
                scores.append((score, idx))

        scores.sort(reverse=True, key=lambda x: x[0])
        return [self._result(self.papers[idx], score, rank + 1) for rank, (score, idx) in enumerate(scores[:top_k])]

    def _result(self, paper, score, rank):
        result = dict(paper)
        result.pop("search_text", None)
        result["score"] = round(score, 3)
        result["rank"] = rank
        return result

    def get_paper(self, paper_id):
        for paper in self.papers:
            if paper.get("id") == paper_id:
                result = dict(paper)
                result.pop("search_text", None)
                return result
        return None


def common_items(results, key, limit=6):
    counter = Counter()
    for paper in results:
        values = paper.get(key, [])
        if isinstance(values, str):
            values = [values]
        counter.update(v for v in values if v)
    return [item for item, _ in counter.most_common(limit)]


def detect_intent(question):
    q = question.lower()
    if any(x in question for x in ["缺点", "不足", "弱点", "局限"]) or any(x in q for x in ["weakness", "limitation"]):
        return "weakness"
    if any(x in question for x in ["优点", "贡献", "亮点"]) or any(x in q for x in ["strength", "contribution"]):
        return "strength"
    if any(x in question for x in ["评分", "评审", "review", "rating"]):
        return "review"
    if any(x in question for x in ["推荐", "有哪些", "找", "相关"]):
        return "recommend"
    return "general"


def local_answer(question, results, stats):
    if not results:
        return "没有检索到足够相关的论文。可以换一个关键词，例如 LLM、reinforcement learning、multimodal、safety 或 diffusion。"

    intent = detect_intent(question)
    areas = common_items(results, "primary_area", 4)
    keywords = common_items(results, "keywords", 8)
    lines = []
    lines.append(f"根据本地 ICLR 2026 索引，我找到 {len(results)} 篇最相关论文。")
    if areas or keywords:
        area_text = "、".join(areas[:3]) if areas else "暂无"
        kw_text = "、".join(keywords[:6]) if keywords else "暂无"
        lines.append(f"这些结果主要集中在 {area_text}；高频关键词包括 {kw_text}。")

    if intent == "weakness":
        lines.append("")
        lines.append("评审中反复出现的不足可以这样归纳：")
        for paper in results[:5]:
            weaknesses = paper.get("reviews", {}).get("weaknesses", [])
            if weaknesses:
                lines.append(f"《{paper['title']}》：{clip(weaknesses[0], 190)}")
            else:
                lines.append(f"《{paper['title']}》：索引中没有公开 weakness 字段，主要依据摘要判断相关。")
    elif intent in {"strength", "review"}:
        lines.append("")
        lines.append("从评审和摘要看，重点证据如下：")
        for paper in results[:5]:
            review = paper.get("reviews", {})
            rating = review.get("avg_rating")
            strength = review.get("strengths", [""])[0] if review.get("strengths") else ""
            rating_text = f"平均评分 {rating}" if rating is not None else "暂无评分"
            lines.append(f"《{paper['title']}》：{rating_text}。{clip(strength or paper.get('abstract', ''), 190)}")
    else:
        lines.append("")
        lines.append("可优先阅读这些论文：")
        for paper in results[:5]:
            review = paper.get("reviews", {})
            rating = review.get("avg_rating")
            rating_text = f"，平均评分 {rating}" if rating is not None else ""
            kw = "、".join(paper.get("keywords", [])[:3])
            lines.append(
                f"《{paper['title']}》{rating_text}。"
                f"方向：{paper.get('primary_area') or '未知'}；关键词：{kw or '无'}。"
                f"{clip(paper.get('abstract', ''), 180)}"
            )

    lines.append("")
    lines.append("建议展示时可以说：系统先用问题检索论文摘要、关键词和评审意见，再把相关论文作为证据生成回答，同时把论文、作者、关键词和研究领域连成一个小型知识图谱。")
    return "\n".join(lines)


def get_deepseek_config():
    api_key = os.environ.get("AIX_LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    base_url = os.environ.get("AIX_LLM_BASE_URL") or os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
    model = os.environ.get("AIX_LLM_MODEL") or os.environ.get("DEEPSEEK_MODEL") or "deepseek-v4-flash"
    return api_key, base_url.rstrip("/"), model


def compact_papers(results, limit=8):
    context = []
    for paper in results[:limit]:
        review = paper.get("reviews", {})
        context.append({
            "id": paper.get("id"),
            "title": paper.get("title"),
            "authors": paper.get("authors", [])[:8],
            "venue": paper.get("venue"),
            "status": paper.get("status"),
            "area": paper.get("primary_area"),
            "keywords": paper.get("keywords", [])[:10],
            "abstract": paper.get("abstract"),
            "avg_rating": review.get("avg_rating"),
            "avg_confidence": review.get("avg_confidence"),
            "review_count": review.get("review_count"),
            "meta_review": review.get("meta_review"),
            "strengths": review.get("strengths", [])[:2],
            "weaknesses": review.get("weaknesses", [])[:2],
            "questions": review.get("questions", [])[:2],
            "comments": review.get("comments", [])[:2],
            "openreview_url": paper.get("openreview_url"),
            "pdf": paper.get("pdf"),
        })
    return context


def deepseek_chat(messages, temperature=0.2, timeout=90):
    api_key, base_url, model = get_deepseek_config()
    if not api_key:
        return None

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }

    url = base_url.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
    except (urllib.error.URLError, KeyError, json.JSONDecodeError, TimeoutError) as exc:
        print(f"DeepSeek request failed: {exc}")
        return None


def call_llm(question, results, history=None, instruction=None):
    system_prompt = instruction or (
        "你是一个顶会论文调研助手。只能依据给定 ICLR 2026 检索结果回答，不要编造不存在的论文。"
        "回答必须使用中文，并用论文标题或短标题说明依据，不要用 [1]、[2] 这样的序号引用。"
        "回答结构：先直接回答，再按研究方向分类，再列代表论文，最后总结趋势和不足。"
    )
    context = compact_papers(results, limit=8)
    history_text = json.dumps(history or [], ensure_ascii=False)[:3500]
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"用户问题：{question}\n\n"
                f"最近对话历史：{history_text}\n\n"
                f"检索到的论文证据 JSON：{json.dumps(context, ensure_ascii=False)}"
            ),
        },
    ]
    return deepseek_chat(messages)


def build_graph(results):
    nodes = {}
    links = []

    def node(node_id, label, kind, weight=1):
        if node_id not in nodes:
            nodes[node_id] = {"id": node_id, "label": label, "kind": kind, "weight": weight}
        else:
            nodes[node_id]["weight"] += weight

    for paper in results[:8]:
        pid = "p:" + paper["id"]
        node(pid, paper["title"], "paper", max(1, 9 - paper["rank"]))
        if paper.get("primary_area"):
            aid = "area:" + paper["primary_area"]
            node(aid, paper["primary_area"], "area", 2)
            links.append({"source": pid, "target": aid, "label": "area"})
        for keyword in paper.get("keywords", [])[:4]:
            kid = "kw:" + keyword.lower()
            node(kid, keyword, "keyword", 2)
            links.append({"source": pid, "target": kid, "label": "keyword"})
        for author in paper.get("authors", [])[:2]:
            auid = "author:" + author
            node(auid, author, "author", 1)
            links.append({"source": pid, "target": auid, "label": "author"})

    return {"nodes": list(nodes.values()), "links": links}


def json_response(handler, payload, status=200):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def make_handler(engine):
    class AppHandler(BaseHTTPRequestHandler):
        server_version = "AIXPaperQA/1.0"

        def log_message(self, fmt, *args):
            print("%s - %s" % (self.address_string(), fmt % args))

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/api/stats":
                json_response(self, {
                    "metadata": engine.metadata,
                    "stats": engine.stats,
                    "areas": [x[0] for x in engine.stats.get("top_areas", [])],
                    "llm_configured": bool(os.environ.get("AIX_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")),
                })
                return
            if parsed.path == "/api/search":
                params = urllib.parse.parse_qs(parsed.query)
                query = params.get("q", [""])[0]
                top_k = int(params.get("top_k", ["8"])[0])
                results = engine.search(query, top_k=top_k)
                json_response(self, {"query": query, "results": results, "graph": build_graph(results)})
                return
            self.serve_static(parsed.path)

        def do_POST(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != "/api/ask":
                json_response(self, {"error": "not found"}, 404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            try:
                body = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                json_response(self, {"error": "invalid json"}, 400)
                return

            question = str(body.get("question", "")).strip()
            top_k = int(body.get("top_k", 8))
            filters = {}
            if body.get("area"):
                filters["area"] = body["area"]
            if body.get("status"):
                filters["status"] = body["status"]

            results = engine.search(question, top_k=top_k, filters=filters)
            answer = call_llm(question, results) if body.get("use_llm") else None
            answer_mode = "llm" if answer else "local_rag"
            if not answer:
                answer = local_answer(question, results, engine.stats)
            json_response(self, {
                "question": question,
                "answer": answer,
                "answer_mode": answer_mode,
                "results": results,
                "graph": build_graph(results),
            })

        def serve_static(self, path):
            if path in {"", "/"}:
                path = "/index.html"
            relative = Path(path.lstrip("/"))
            target = (WEB_DIR / relative).resolve()
            web_root = WEB_DIR.resolve()
            if web_root not in target.parents and target != web_root:
                json_response(self, {"error": "forbidden"}, 403)
                return
            if not target.exists() or not target.is_file():
                json_response(self, {"error": "not found"}, 404)
                return
            ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            body = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith("text/") or ctype == "application/javascript" else ""))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return AppHandler


def main():
    load_env_file()
    parser = argparse.ArgumentParser(description="Run the local ICLR 2026 RAG QA web app.")
    parser.add_argument("--index", type=Path, default=INDEX_FILE)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if not args.index.exists():
        raise FileNotFoundError(f"Index not found: {args.index}. Run: python build_index.py")

    engine = SearchEngine(args.index)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    print(f"ICLR 2026 Paper QA is running at http://{args.host}:{args.port}")
    print(f"Loaded {len(engine.papers)} papers from {args.index}")
    server.serve_forever()


if __name__ == "__main__":
    main()
