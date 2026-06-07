# 顶会论文检索系统

一个面向 ICLR 2026 OpenReview 论文数据的 **LLM + RAG + 知识图谱论文调研系统**。系统支持用自然语言检索顶会论文、查看论文详情、调用 DeepSeek 做论文问答、生成论文思维导图、横向比较论文，并基于会议方向生成局部知识图谱和研究趋势分析。

项目适合作为 AI 通识课、AI + X 创意项目、文献调研工具或课程展示 Demo。

## 项目定位

顶会论文数量庞大、研究方向分散，普通学生在做文献调研时经常遇到这些问题：

- 不知道某个方向有哪些相关论文。
- 只看标题和摘要，很难快速理解论文贡献。
- 多篇论文之间缺少横向比较。
- 很难从 reviewer 意见里总结论文优缺点。
- 不知道一个方向的发展趋势和可继续探索的问题。

本项目希望把论文数据库变成一个可交互的 AI 调研助手：

```text
用户自然语言问题
        ↓
本地论文索引检索
        ↓
取出标题、摘要、关键词、评审意见等证据
        ↓
DeepSeek 基于证据生成回答
        ↓
展示来源论文、论文卡片、详情、图谱和趋势分析
```

系统不是简单的关键词搜索，而是尽量让用户像和科研助手对话一样完成文献调研。

## 核心功能

### 1. 论文调研工作台

主页面由三部分组成：

- 左上：自然语言检索输入框。
- 左下：检索结果列表。
- 中间：选中论文详情。
- 右侧：AI 助手对话框。

支持输入类似问题：

```text
我想找和大语言模型、检索增强、知识图谱有关，适合做课程展示的论文
```

系统会先用 DeepSeek 做检索意图规划，把中文需求改写成适合论文索引检索的英文关键词组合，然后在本地 ICLR 2026 索引中检索相关论文。

检索结果会展示：

- 论文标题
- 作者
- 自动分类
- 原始会议方向
- 评分
- 相似度
- 摘要片段
- OpenReview 链接
- 加入对比按钮

### 2. RAG 问答助手

右侧 AI 助手支持多轮问答。回答范围有三种：

- 当前检索结果
- 选中论文
- 全库重新检索

典型问题：

```text
这些论文主要分成哪几类？
哪几篇最适合初学者做课程汇报？
这篇论文的主要贡献和局限是什么？
这些 RAG 论文和知识图谱有什么关系？
```

问答流程：

```text
用户问题
  ↓
根据回答范围检索相关论文
  ↓
抽取论文标题、摘要、关键词、评审意见
  ↓
DeepSeek 生成回答
  ↓
保存来源论文
```

每个会话可以新建、删除，历史对话会在固定区域滚动显示。

### 3. 论文详情

点击论文后可以查看：

- 标题
- 作者
- 自动分类
- 评分
- 评审数量
- 关键词
- 摘要
- Reviewer 肯定点
- Reviewer 质疑点
- Meta Review
- OpenReview 链接
- PDF 链接

详情页内置三个标签：

```text
摘要 / 思维导图 / 评审
```

### 4. 论文思维导图

在论文详情页点击 `生成论文思维导图` 后，系统会调用 DeepSeek 生成结构化思维导图数据，并用 Markmap 渲染成交互式思维导图。

思维导图包含：

- 研究问题
- 核心思路
- 方法流程
- 实验评估
- 主要贡献
- 局限与追问

说明：

- 思维导图不会默认生成，只有点击按钮后才调用 DeepSeek。
- Markmap 通过 CDN 加载，展示机器需要能访问网络。
- 如果 DeepSeek 返回格式不完整，页面会显示生成失败提示。

### 5. 论文对比

论文对比页面支持：

- 从上一轮检索结果选择论文加入对比。
- 让系统根据对比目标自动推荐论文。
- 生成基础对比表。
- 调用 DeepSeek 生成完整横向对比。

对比维度包括：

- 研究问题
- 自动分类
- 研究方向
- 方法线索
- 数据集线索
- 评分
- 常见不足线索

示例对比目标：

```text
帮我挑几篇方法路线不同、适合课堂横向比较的论文
```

### 6. 论文库浏览

论文库页面支持：

- 全库分页浏览
- 标题搜索
- 作者搜索
- 关键词搜索
- 方向搜索
- 每页显示 12 / 24 / 36 / 50 篇
- 弹窗查看论文详情
- 加入对比

论文库的“查看详情”不会跳回主页面，而是在当前页面弹窗展示。

### 7. 研究图谱

研究图谱页以 ICLR 2026 的会议方向为主题入口。会议方向来自论文数据中的 `primary_area` 字段，也就是侧边栏 `方向筛选` 使用的同一套方向。

例如：

```text
foundation or frontier models, including LLMs
generative models
reinforcement learning
learning on graphs and other geometries & topologies
alignment, fairness, safety, privacy, and societal considerations
```

可以额外输入细化主题：

```text
retrieval
graph learning
tool use
diffusion
```

生成后会展示：

- 当前会议方向总论文数
- 当前筛选论文数
- 论文节点
- 方向节点
- 方法节点
- 数据集节点
- 关键词节点
- 论文之间的相似关系
- DeepSeek 研究趋势分析

图谱数量与侧边栏 `检索论文数量` 保持一致。

## 技术架构

### 技术栈

| 模块 | 技术 |
|---|---|
| 前端展示 | Streamlit |
| 本地检索 | BM25 风格关键词检索 |
| LLM | DeepSeek Chat Completions API |
| RAG | 本地检索结果 + DeepSeek 生成 |
| 知识图谱 | PyVis |
| 思维导图 | Markmap |
| 数据处理 | Python / JSONL |
| 表格处理 | pandas |

### 系统流程

```text
2026/submissions.jsonl
        ↓
build_index.py
        ↓
data/iclr2026_index.json
        ↓
app.py / streamlit_app.py
        ↓
SearchEngine 本地检索
        ↓
DeepSeek RAG 回答 / 对比 / 趋势 / 思维导图
        ↓
Streamlit 页面展示
```

### 主要文件

```text
.
├── app.py                  # 本地检索引擎、DeepSeek 调用、轻量 HTTP API
├── build_index.py           # 从 OpenReview JSONL 构建紧凑索引
├── streamlit_app.py         # Streamlit 主应用
├── start.ps1                # Windows 一键启动脚本
├── requirements.txt         # Python 依赖
├── .streamlit/config.toml   # Streamlit 本地配置
├── web/                     # 早期轻量 Web 前端资源
├── 2026/                    # ICLR 2026 原始数据目录，需要自行放置
└── data/
    └── iclr2026_index.json  # 构建后的本地索引，不建议提交到 Git
```

## 数据说明

项目默认读取：

```text
2026/*/submissions.jsonl
```

`build_index.py` 会自动选择更大的 `submissions.jsonl`，因为较大的文件通常包含更完整的评审、评论和决策信息。

当前开发环境中的数据规模：

```text
accepted papers: 5354
papers with reviews: 5354
index file: data/iclr2026_index.json
```

索引中使用的主要字段：

- `title`
- `authors`
- `abstract`
- `keywords`
- `primary_area`
- `venue`
- `status`
- `openreview_url`
- `pdf`
- `reviews.avg_rating`
- `reviews.avg_confidence`
- `reviews.strengths`
- `reviews.weaknesses`
- `reviews.questions`
- `reviews.meta_review`
- `reviews.comments`

注意：

- 原始数据目录 `2026/` 通常较大，不建议提交到 GitHub。
- 构建后的 `data/iclr2026_index.json` 也较大，本项目 `.gitignore` 默认忽略 `data/*.json`。
- 其他人复现时需要自己准备同结构的 OpenReview 数据。

## 环境安装

推荐 Python 版本：

```text
Python 3.10
```

### 方式一：Conda

创建环境：

```powershell
conda create -n aix-rag python=3.10 -y
```

安装依赖：

```powershell
conda run -n aix-rag pip install -r requirements.txt
```

或者先激活环境：

```powershell
conda activate aix-rag
pip install -r requirements.txt
```

### 方式二：venv

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## DeepSeek 配置

在项目根目录创建 `.env` 文件：

```text
AIX_LLM_API_KEY=你的 DeepSeek API Key
AIX_LLM_BASE_URL=https://api.deepseek.com
AIX_LLM_MODEL=deepseek-v4-flash
AIX_DISABLE_PROXY=1
```

也兼容这些变量名：

```text
DEEPSEEK_API_KEY
DEEPSEEK_BASE_URL
DEEPSEEK_MODEL
```

不要把 `.env` 提交到 GitHub。当前 `.gitignore` 已经忽略 `.env`。

如果你本地代理环境变量导致 DeepSeek 请求失败，可以保留：

```text
AIX_DISABLE_PROXY=1
```

它会在程序启动时移除 `http_proxy` / `https_proxy` / `HTTP_PROXY` / `HTTPS_PROXY`。

## 构建索引

确保目录结构类似：

```text
2026/
├── xxxx-xxxx/
│   ├── manifest.json
│   └── submissions.jsonl
└── yyyy-yyyy/
    ├── manifest.json
    └── submissions.jsonl
```

构建默认索引：

```powershell
conda run -n aix-rag python build_index.py
```

默认行为：

- 数据目录：`2026/`
- 输出文件：`data/iclr2026_index.json`
- 数据来源：自动选择更大的 `submissions.jsonl`
- 论文状态：只保留 `accepted`

常用参数：

```powershell
# 使用指定数据目录
conda run -n aix-rag python build_index.py --data-dir 2026

# 输出到指定文件
conda run -n aix-rag python build_index.py --out data/iclr2026_index.json

# 使用含评审的完整数据
conda run -n aix-rag python build_index.py --source reviews

# 使用较小的 paper-only 数据
conda run -n aix-rag python build_index.py --source small

# 保留所有状态论文
conda run -n aix-rag python build_index.py --status all

# 构建一个小型测试索引
conda run -n aix-rag python build_index.py --limit 100
```

构建成功后会看到类似输出：

```text
wrote data/iclr2026_index.json
papers: 5354, with reviews: 5354
top areas:
  foundation or frontier models, including LLMs: 845
  applications to computer vision, audio, language, and other modalities: 733
  generative models: 498
```

## 启动系统

### Windows 一键启动

```powershell
.\start.ps1
```

`start.ps1` 会自动：

1. 切换到项目目录。
2. 清理代理环境变量。
3. 如果索引不存在，自动运行 `build_index.py`。
4. 启动 Streamlit。

浏览器打开：

```text
http://127.0.0.1:8501
```

### 手动启动

```powershell
conda run -n aix-rag streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

如果不用 Conda：

```powershell
streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

### Streamlit 配置

`.streamlit/config.toml` 中已经配置：

```toml
[browser]
gatherUsageStats = false

[server]
headless = true
address = "127.0.0.1"
port = 8501
```

## 使用指南

### 1. 进入论文调研工作台

默认进入：

```text
论文调研工作台
```

推荐流程：

1. 在左上角输入自然语言需求。
2. 点击 `AI 检索`。
3. 在左下角浏览论文列表。
4. 点击某篇论文的 `选择`。
5. 在中间查看论文详情。
6. 在右侧 AI 助手继续追问。

### 2. 使用侧边栏筛选

侧边栏默认展开 `检索筛选`：

- 方向筛选
- 接收类型
- 额外关键词
- 检索论文数量
- 追问优先使用上一轮论文

`检索论文数量` 范围：

```text
3 - 100
```

这个数量会影响：

- 主工作台检索结果数量
- AI 助手重新检索数量
- 研究图谱筛选数量

### 3. AI 助手追问

右侧 AI 助手的问题框默认为空。

发送问题后：

- 问题会进入当前会话。
- DeepSeek 会基于论文证据回答。
- 输入框会自动清空。
- 来源论文会记录在当前会话中。

### 4. 查看论文思维导图

在论文详情中打开：

```text
思维导图
```

点击：

```text
生成论文思维导图
```

系统会调用 DeepSeek 生成结构化导图，并用 Markmap 展示。

### 5. 论文对比

进入：

```text
论文对比
```

可以：

- 使用 AI 推荐论文加入对比。
- 从上一轮检索结果手动选择论文。
- 查看基础表格。
- 生成 DeepSeek 完整对比表。

### 6. 论文库与研究图谱

进入：

```text
论文库与研究图谱
```

包含两个标签：

```text
论文库浏览 / 研究图谱
```

论文库浏览支持分页搜索。

研究图谱以 ICLR 2026 会议方向为入口：

1. 选择会议方向。
2. 可选填细化主题。
3. 点击生成知识图谱和研究趋势分析。

## 复现步骤

如果其他人在新电脑上复现，按这个顺序做：

1. 克隆仓库。

```bash
git clone <your-repo-url>
cd paper-graph-rag-assistant
```

2. 创建 Python 环境。

```powershell
conda create -n aix-rag python=3.10 -y
conda run -n aix-rag pip install -r requirements.txt
```

3. 准备数据。

把 ICLR 2026 OpenReview 数据放到：

```text
2026/*/submissions.jsonl
```

4. 配置 `.env`。

```text
AIX_LLM_API_KEY=你的 DeepSeek API Key
AIX_LLM_BASE_URL=https://api.deepseek.com
AIX_LLM_MODEL=deepseek-v4-flash
AIX_DISABLE_PROXY=1
```

5. 构建索引。

```powershell
conda run -n aix-rag python build_index.py
```

6. 启动应用。

```powershell
conda run -n aix-rag streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

7. 打开浏览器。

```text
http://127.0.0.1:8501
```

## 常见问题

### 1. 启动时报 `Index not found`

说明还没有构建索引。

执行：

```powershell
conda run -n aix-rag python build_index.py
```

### 2. `Data directory not found: 2026`

说明没有放原始数据目录。

需要保证存在：

```text
2026/*/submissions.jsonl
```

### 3. DeepSeek 没有返回结果

检查 `.env`：

```text
AIX_LLM_API_KEY=你的 DeepSeek API Key
AIX_LLM_BASE_URL=https://api.deepseek.com
AIX_LLM_MODEL=deepseek-v4-flash
```

如果你本地有代理，但代理端口不可用，保留：

```text
AIX_DISABLE_PROXY=1
```

### 4. Markmap 思维导图不显示

思维导图依赖 Markmap CDN：

```text
https://cdn.jsdelivr.net/npm/markmap-autoloader@latest
```

如果展示电脑不能访问外网，Markmap 可能无法加载。

### 5. 论文很多时图谱比较卡

侧边栏 `检索论文数量` 会直接决定图谱节点数量。数量越大，图谱越复杂。

建议展示时使用：

```text
20 - 50
```

如果只是搜索论文，可以调到 100。

### 6. Conda 命令不可用

可以改用 venv，或重新初始化 Conda：

```powershell
conda init powershell
```

然后重启 PowerShell。

## Git 提交建议

建议提交：

```text
app.py
build_index.py
streamlit_app.py
requirements.txt
start.ps1
README.md
.streamlit/config.toml
.gitignore
web/
```

不建议提交：

```text
.env
2026/
data/iclr2026_index.json
data/*.log
__pycache__/
```

当前 `.gitignore` 已经包含：

```text
.env
data/*.json
data/*.log
__pycache__/
*.pyc
```

如果要把原始数据也排除，可以额外加入：

```text
2026/
```

## 汇报建议

汇报时不要把项目说成“论文搜索系统”，可以这样表述：

> 本项目构建了一个面向 ICLR 2026 顶会论文数据的 AI 调研助手。系统通过 RAG 机制约束大模型回答，使回答基于真实论文摘要、关键词和评审意见生成，并提供论文卡片、论文详情、思维导图、横向对比、知识图谱和趋势分析，帮助科研新手快速完成顶会论文调研。

可以重点展示：

1. 用自然语言找论文。
2. AI 助手基于来源论文回答。
3. 点击论文查看详情和 reviewer 优缺点。
4. 一键生成论文思维导图。
5. 多篇论文自动对比。
6. 按会议方向生成知识图谱和研究趋势。

## 项目名称建议

中文名：

```text
顶会论文检索系统
```

更完整的论文式名称：

```text
面向 ICLR 2026 的 LLM + RAG 增强知识图谱问答系统
```

GitHub 仓库名建议：

```text
paper-graph-rag-assistant
```
