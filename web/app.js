const state = {
  stats: null,
  lastResults: [],
};

const $ = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function clip(value, limit) {
  const text = String(value || "").replace(/\s+/g, " ").trim();
  return text.length > limit ? `${text.slice(0, limit - 1).trim()}...` : text;
}

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

function renderStats(data) {
  state.stats = data;
  $("paperCount").textContent = data.stats.paper_count ?? "--";
  $("reviewCount").textContent = data.stats.papers_with_reviews ?? "--";
  $("llmToggle").disabled = !data.llm_configured;
  $("llmToggle").title = data.llm_configured ? "已检测到模型配置" : "未检测到外部模型配置";

  const areaSelect = $("areaSelect");
  for (const area of data.areas || []) {
    const option = document.createElement("option");
    option.value = area;
    option.textContent = area;
    areaSelect.appendChild(option);
  }

  const topics = [
    "大语言模型与 RAG",
    "多模态推理",
    "强化学习探索",
    "AI 安全与对齐",
    "扩散模型",
    "机器人与具身智能",
  ];
  $("topicList").innerHTML = topics
    .map((topic) => `<button class="topic-chip" type="button">${escapeHtml(topic)}</button>`)
    .join("");
  document.querySelectorAll(".topic-chip").forEach((button) => {
    button.addEventListener("click", () => {
      $("questionInput").value = `请推荐 ICLR 2026 中关于${button.textContent}的论文，并说明理由`;
      ask();
    });
  });
}

function renderAnswer(data) {
  $("answerMode").textContent = data.answer_mode === "llm" ? "外部 LLM" : "本地 RAG";
  $("answerBox").classList.remove("loading");
  $("answerBox").textContent = data.answer || "";
}

function renderResults(results) {
  state.lastResults = results;
  $("resultCount").textContent = `${results.length} 条`;
  if (!results.length) {
    $("resultsList").innerHTML = `<div class="paper-card">没有匹配结果</div>`;
    return;
  }

  $("resultsList").innerHTML = results.map((paper) => {
    const review = paper.reviews || {};
    const rating = review.avg_rating != null ? `<span class="badge rating">评分 ${escapeHtml(review.avg_rating)}</span>` : "";
    const area = paper.primary_area ? `<span class="badge area">${escapeHtml(paper.primary_area)}</span>` : "";
    const keywords = (paper.keywords || []).slice(0, 4)
      .map((keyword) => `<span class="badge">${escapeHtml(keyword)}</span>`)
      .join("");
    const pdf = paper.pdf ? `<a href="https://openreview.net${escapeHtml(paper.pdf)}" target="_blank" rel="noreferrer">PDF</a>` : "";
    return `
      <article class="paper-card">
        <h3><a href="${escapeHtml(paper.openreview_url)}" target="_blank" rel="noreferrer">${escapeHtml(paper.title)}</a></h3>
        <div class="meta">${escapeHtml((paper.authors || []).slice(0, 6).join(", "))}</div>
        <p class="abstract">${escapeHtml(clip(paper.abstract, 280))}</p>
        <div class="badges">${rating}${area}${keywords}</div>
        <div class="card-actions">
          <a href="${escapeHtml(paper.openreview_url)}" target="_blank" rel="noreferrer">OpenReview</a>
          ${pdf}
        </div>
      </article>
    `;
  }).join("");
}

function colorForKind(kind) {
  return {
    paper: "#2563eb",
    keyword: "#0f766e",
    area: "#b7791f",
    author: "#7c3aed",
  }[kind] || "#647084";
}

function renderGraph(graph) {
  const svg = $("graphSvg");
  const nodes = graph.nodes || [];
  const links = graph.links || [];
  $("graphCount").textContent = `${nodes.length} 点 ${links.length} 边`;
  svg.innerHTML = "";
  if (!nodes.length) return;

  const width = svg.clientWidth || 480;
  const height = svg.clientHeight || 360;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

  const paperNodes = nodes.filter((node) => node.kind === "paper");
  const otherNodes = nodes.filter((node) => node.kind !== "paper");
  const positions = new Map();
  const cx = width / 2;
  const cy = height / 2;
  const inner = Math.max(72, Math.min(width, height) * 0.24);
  const outer = Math.max(126, Math.min(width, height) * 0.41);

  paperNodes.forEach((node, index) => {
    const angle = (Math.PI * 2 * index) / Math.max(paperNodes.length, 1) - Math.PI / 2;
    positions.set(node.id, {
      x: cx + Math.cos(angle) * inner,
      y: cy + Math.sin(angle) * inner,
    });
  });

  otherNodes.forEach((node, index) => {
    const angle = (Math.PI * 2 * index) / Math.max(otherNodes.length, 1) - Math.PI / 2;
    positions.set(node.id, {
      x: cx + Math.cos(angle) * outer,
      y: cy + Math.sin(angle) * outer,
    });
  });

  const linkLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
  const nodeLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
  svg.append(linkLayer, nodeLayer);

  for (const link of links) {
    const source = positions.get(link.source);
    const target = positions.get(link.target);
    if (!source || !target) continue;
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", source.x);
    line.setAttribute("y1", source.y);
    line.setAttribute("x2", target.x);
    line.setAttribute("y2", target.y);
    line.setAttribute("stroke", "#c9d3df");
    line.setAttribute("stroke-width", "1.2");
    linkLayer.appendChild(line);
  }

  for (const node of nodes) {
    const pos = positions.get(node.id);
    if (!pos) continue;
    const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
    const radius = node.kind === "paper" ? 8 + Math.min(node.weight, 6) : 6 + Math.min(node.weight, 4);

    group.setAttribute("transform", `translate(${pos.x}, ${pos.y})`);
    circle.setAttribute("r", radius);
    circle.setAttribute("fill", colorForKind(node.kind));
    circle.setAttribute("opacity", node.kind === "paper" ? "0.92" : "0.78");
    label.setAttribute("class", "node-label");
    label.setAttribute("x", radius + 4);
    label.setAttribute("y", "4");
    label.textContent = clip(node.label, node.kind === "paper" ? 24 : 18);
    title.textContent = node.label;
    group.append(circle, label, title);
    nodeLayer.appendChild(group);
  }
}

async function ask() {
  const question = $("questionInput").value.trim();
  if (!question) return;
  $("askButton").disabled = true;
  $("answerBox").classList.add("loading");
  $("answerBox").textContent = "检索中...";
  try {
    const data = await postJson("/api/ask", {
      question,
      top_k: 8,
      area: $("areaSelect").value,
      use_llm: $("llmToggle").checked,
    });
    renderAnswer(data);
    renderResults(data.results || []);
    renderGraph(data.graph || { nodes: [], links: [] });
  } catch (error) {
    $("answerBox").classList.remove("loading");
    $("answerBox").textContent = `请求失败：${error.message}`;
  } finally {
    $("askButton").disabled = false;
  }
}

async function init() {
  try {
    const stats = await getJson("/api/stats");
    renderStats(stats);
    await ask();
  } catch (error) {
    $("answerBox").textContent = `应用初始化失败：${error.message}`;
  }
}

$("askButton").addEventListener("click", ask);
$("questionInput").addEventListener("keydown", (event) => {
  if (event.key === "Enter") ask();
});
$("areaSelect").addEventListener("change", ask);

init();
