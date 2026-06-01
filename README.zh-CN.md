<div align="center">
  <img
    src="https://github.com/user-attachments/assets/aa171a4c-074c-4082-b3d1-c70f5f7f2aca"
    alt="XMem Logo"
    width="100%"
  />
</div>

<div align="center">
  <h1>XMem</h1>
  <p><strong>永不遗忘的 AI 记忆层</strong></p>
  <p>为每一个 AI Agent 和 LLM 界面开箱即用地提供持久、跨平台记忆。</p>

<img src="https://img.shields.io/badge/python-3.11+-blue?logo=python&logoColor=white" alt="Python 3.11+"/>
<img src="https://img.shields.io/badge/license-BSD--3--Clause-green" alt="BSD-3 License"/>
<img src="https://img.shields.io/badge/FastAPI-00C7B7?logo=fastapi&logoColor=white" alt="FastAPI"/>
<br/>
<img src="https://img.shields.io/badge/LangGraph-6C47FF?logo=langchain&logoColor=white" alt="LangGraph"/>
<img src="https://img.shields.io/badge/Rust-Weaver-b7410e?logo=rust&logoColor=white" alt="Rust Weaver"/>
<img src="https://img.shields.io/badge/Multi--LLM-Gemini%20%7C%20Claude%20%7C%20GPT%20%7C%20Bedrock%20%7C%20Ollama-orange" alt="Multi-LLM"/>
</div>

<hr>

<p align="center">
  <a href="README.md">English</a> &nbsp;&bull;&nbsp;
  <a href="README.zh-CN.md">简体中文</a> &nbsp;&bull;&nbsp;
  <a href="README.ja.md">日本語</a>
</p>

<p align="center">
  <a href="#演示">演示</a> &nbsp;&bull;&nbsp;
  <a href="#功能">功能</a> &nbsp;&bull;&nbsp;
  <a href="#架构">架构</a> &nbsp;&bull;&nbsp;
  <a href="#基准测试">基准测试</a> &nbsp;&bull;&nbsp;
  <a href="#快速开始">快速开始</a> &nbsp;&bull;&nbsp;
  <a href="#配置">配置</a>
</p>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=XortexAI/XMem&type=date&theme=dark&legend=top-left" />
  <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=XortexAI/XMem&type=date&legend=top-left" />
  <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=XortexAI/XMem&type=date&legend=top-left" />
</picture>

## 更新 / 新闻
- **[2026 年 6 月 1 日]** XMem 现在拥有原生 Golang 版本的记忆层。它面向更高吞吐、更低延迟和生产级部署而构建，适用于需要在数百万次交互中可靠运行记忆能力的场景。
- **[2026 年 5 月 25 日]** 本地工作区支持现已上线。只需 3 条命令即可在本地设置 XMem，并在几分钟内开始构建带记忆的应用。设置说明见 [Local.md](https://github.com/XortexAI/XMem/blob/main/Local.md)。
 ```bash
npx create-xmem@latest
cd xmem
npm run dev
```

## 什么是 XMem？

每一次与 LLM 的对话都像从零开始。切换工具、切换模型提供商，或者下周再回来，所有上下文都会消失。

XMem 是印度排名第一的开源 Agentic Memory Layer。我们正在推出 Memory-as-a-Service，也就是面向每一种 AI 用例和领域的记忆层：无论是长时间运行 Agent 的时间记忆、患者上下文的医疗记忆、团队和项目的企业记忆，还是编码 Agent 与工作流的开发者记忆。

这是一个首创的、面向有状态 AI 的 Agentic Memory Layer。
与只存储和检索片段的传统记忆系统不同，XMem 将记忆变成一个主动推理过程。它会决定该记住什么、更新什么、忘记什么，并将每条记忆动态路由到正确的专用 Agent 和存储系统。

## 演示

只需在你选择的任意 AI 平台中输入 “X”，即可在 XMem 提供的四种模式之间切换，无缝存储和搜索记忆、从已有聊天导入上下文，或使用已索引的代码仓库。

https://github.com/user-attachments/assets/8e3349ab-63c9-4046-821d-ca8097948440

## 功能

### Chrome 扩展

XMem Chrome 扩展为 ChatGPT、Claude、Gemini、DeepSeek 和 Perplexity 带来持久记忆。

**实时搜索与注入** - 当你输入提示词时，XMem 会实时搜索你的记忆并显示一个悬浮标签。点击一次即可将相关上下文直接注入输入框，零摩擦。

**后台自动保存（Xingest）** - 当你点击“发送”时，XMem 会异步捕获这一轮对话。后台队列会提取事实和摘要，不影响你的界面。

https://github.com/user-attachments/assets/97793cf9-d247-4d02-9c31-3cc9bbbf89aa

### Agent 插件

新的 [`plugin/`](plugin/) 文件夹将 XMem 直接带入开发者 Agent 和编码助手。它包含 Claude Code、Codex、Cursor、Hermes、OpenClaw 和 OpenCode 的第一方集成，让 Agent 可以搜索已有记忆、保存持久项目知识，并在不同会话之间延续上下文，同时将 API Key 保存在环境变量或客户端专用的密钥存储中。

### Context

Context 让你无需手动复制粘贴，就能把已有对话带入 XMem。

粘贴一个共享的 ChatGPT、Claude 或 Gemini 链接。XMem 会打开它，提取每一条用户和助手消息，并运行完整的 ingest pipeline，让这段对话成为可搜索的记忆。

你也可以上传转录文件（文本、Markdown 或 JSON）。XMem 内置了对 Cursor 和 Antigravity 导出的解析，并会对未知格式使用 LLM 作为兜底解析。

https://github.com/user-attachments/assets/4ff22405-b7ad-4b78-9189-9a6e3ebd5e40

### Scanner

Scanner 会索引整个 Git 仓库，并为你的代码库构建可查询的知识图谱。

索引完成后，你可以用自然语言询问文件、函数、依赖和影响范围。它可用于理解新仓库、查找某个功能的位置、追踪代码连接方式，或判断修改某处会破坏什么。

https://github.com/user-attachments/assets/f0fd393e-3820-404b-8d0e-e452e1dd52d0

### 多领域分类

并非所有记忆都相同，把它们当成同一种数据处理正是其他方案表现不佳的原因。XMem 的 **Classifier Agent** 会分析每一条输入数据，并将其路由到正确的领域：

<table>
  <tr>
    <th>领域</th>
    <th>存储内容</th>
    <th>示例</th>
    <th>存储</th>
  </tr>
  <tr>
    <td><strong>Profile</strong></td>
    <td>永久用户事实、偏好、身份信息</td>
    <td><em>“我更喜欢用 Go 而不是 Python 写后端”</em></td>
    <td>Pinecone</td>
  </tr>
  <tr>
    <td><strong>Temporal</strong></td>
    <td>带日期解析的时间锚定事件</td>
    <td><em>“我昨天晋升为 Staff Engineer”</em></td>
    <td>Neo4j</td>
  </tr>
  <tr>
    <td><strong>Summary</strong></td>
    <td>压缩后的对话要点</td>
    <td><em>“讨论了从 REST 迁移到 gRPC”</em></td>
    <td>Pinecone</td>
  </tr>
  <tr>
    <td><strong>Code</strong></td>
    <td>与符号关联的注释、bug 和解释</td>
    <td><em>“这段重试逻辑存在竞态条件”</em></td>
    <td>Neo4j + Pinecone</td>
  </tr>
  <tr>
    <td><strong>Snippet</strong></td>
    <td>个人代码模式和工具函数</td>
    <td><em>“这是我在 Go 中常用的标准错误处理器”</em></td>
    <td>Pinecone</td>
  </tr>
  <tr>
    <td><strong>Image</strong></td>
    <td>视觉观察和描述</td>
    <td><em>架构图截图</em></td>
    <td>Pinecone</td>
  </tr>
</table>

### Agentic Retrieval

当你查询 XMem 时，检索并不是简单的向量搜索。LLM 自身会决定要查找什么：

1. **工具选择** - 检索 LLM 会分析你的查询，并调用合适的搜索工具（SearchProfile、SearchTemporal、SearchSummary、SearchSnippet），必要时可并行调用多个工具。
2. **综合生成** - 所有搜索工具的结果会被汇总，LLM 会生成带来源引用的答案。

这意味着当你问“我偏好的技术栈是什么，以及我上次重构 auth 模块是什么时候？”时，系统会自动同时触发 profile 查询和 temporal 搜索。

### 多 LLM 编排与兜底

XMem 不绑定某一个模型提供商。它可以在 **Gemini、Claude、OpenAI、OpenRouter、Amazon Bedrock 和 Ollama** 之间编排，并自动故障切换：

```
gemini -> claude -> openai -> bedrock -> ollama
```

如果你的主 LLM 触发限流或发生故障，XMem 会静默切换到下一个提供商。每个 Agent 都可以固定到特定模型，兜底顺序也完全可配置。

### 本地运行

无需云依赖。你可以使用 Ollama 作为 LLM、FastEmbed 作为嵌入模型，并使用 Chroma 或 SQLite 作为向量存储来运行 XMem：

```bash
pip install -e ".[local]"
```

## 架构

<img width="1536" height="1024" alt="WhatsApp Image 2026-04-27 at 11 50 51" src="https://github.com/user-attachments/assets/424d1c77-63e3-48ac-b457-6beecd437f65" />

XMem 是一条由 LangGraph 协调的**专用 AI Agent 管线**，背后由确定性执行层（Weaver）和三个专用存储引擎支撑。

### 写入流程

```
用户输入（SDK / Chrome 扩展 / API）
         |
         v
   +--------------+
   |  Classifier  |    分析文本，并路由到不同领域
   +------+-------+
          |
    +-----+-----+------+----------+
    v     v     v      v          v
 Profile Temporal Summary Code  Snippet     领域 Agent 并行提取结构化数据
 Agent   Agent   Agent  Agent   Agent
    |     |      |      |        |
    v     v      v      v        v
   +----------------------------------+
   |          Judge Agent             |     与现有记忆比较
   |   (ADD / UPDATE / DELETE / NOOP) |     防止重复和过期信息
   +----------------+-----------------+
                    |
                    v
   +----------------------------------+
   |        Weaver（Rust 核心）       |     确定性执行器
   |  Pinecone | Neo4j | MongoDB     |     没有 LLM，纯软件逻辑
   +----------------------------------+
```

1. **Classifier** 将输入路由到相关领域。
2. **Domain Agents**（Profiler、Temporal、Summarizer、Code、Snippet、Image）并行提取结构化数据。
3. **Judge Agent** 将每个提取结果与现有记忆比较，并决定：ADD、UPDATE、DELETE 或 NOOP。
4. **Weaver** 在所有存储后端上确定性执行 Judge 的决策。核心实现为一个独立 Rust crate，不依赖 LLM。

**高努力模式** 会自动将长输入拆分成重叠片段（约 200 token）并行处理，然后合并结果，确保长对话中没有遗漏。

### 检索流程

```
用户查询
    |
    v
+----------------------------------+
|       Retrieval LLM              |
|  决定调用哪些工具：              |
|  SearchProfile, SearchTemporal,  |
|  SearchSummary, SearchSnippet    |
+----------------+-----------------+
                 |
    +------------+------------+
    v            v            v
 Pinecone      Neo4j      Pinecone        并行搜索执行
 (profiles)   (events)   (summaries)
    |            |            |
    +------------+------------+
                 v
+----------------------------------+
|   答案综合 + 引用                |    LLM 生成带来源的答案
+----------------------------------+
```

### 存储

<table>
  <tr>
    <th>引擎</th>
    <th>用途</th>
    <th>用于</th>
  </tr>
  <tr>
    <td><strong>Pinecone</strong></td>
    <td>高速向量相似度搜索</td>
    <td>Profile、摘要、代码片段、代码注释</td>
  </tr>
  <tr>
    <td><strong>Neo4j</strong></td>
    <td>图遍历 + 时间推理</td>
    <td>事件、代码知识图谱、注释</td>
  </tr>
  <tr>
    <td><strong>MongoDB</strong></td>
    <td>原始文档存储</td>
    <td>扫描后的代码、文件元数据、扫描状态</td>
  </tr>
</table>

> [!NOTE]
> 对于本地部署，Pinecone 可以替换为 **Chroma**、**pgvector** 或 **SQLite** 向量存储。

## 基准测试

我们在两个成熟的学术基准上测试了 XMem 和所有主流记忆方案。XMem 在各项指标上都表现领先。

### LoCoMo

测试对记忆的组合推理能力：系统能否连接跨对话事实、推理时间关系，并回答开放式问题？

<table>
  <tr>
    <th>方法</th>
    <th>单跳 (%)</th>
    <th>多跳 (%)</th>
    <th>开放领域 (%)</th>
    <th>时间 (%)</th>
    <th>总体 (%)</th>
  </tr>
  <tr><td><strong>XMEM（我们的）</strong></td><td><strong>90.6</strong></td><td><strong>92.3</strong></td><td><strong>91.2</strong></td><td><strong>91.9</strong></td><td><strong>91.5</strong></td></tr>
  <tr><td>Zep</td><td>74.11</td><td>66.04</td><td>67.71</td><td>79.79</td><td>75.14</td></tr>
  <tr><td>Memobase (v0.0.37)</td><td>70.92</td><td>46.88</td><td>77.17</td><td>85.05</td><td>75.78</td></tr>
  <tr><td>Mem0g (YC 24)</td><td>65.71</td><td>47.19</td><td>75.71</td><td>58.13</td><td>68.44</td></tr>
  <tr><td>Mem0 (YC 24)</td><td>67.13</td><td>51.15</td><td>72.93</td><td>55.51</td><td>66.88</td></tr>
  <tr><td>LangMem</td><td>62.23</td><td>47.92</td><td>71.12</td><td>23.43</td><td>58.10</td></tr>
  <tr><td>OpenAI</td><td>63.79</td><td>42.92</td><td>62.29</td><td>21.71</td><td>52.90</td></tr>
</table>

> 在多跳推理（连接不同对话中的事实）上，XMem 比第二名高出 **26.3 分**。总体上，XMem 以 **91.5%** 领先所有系统，高于 Zep 的 75.14。

### LongMemEval-S

长期对话记忆的行业标准基准。它测试系统能否召回事实、跟踪偏好变化、推理时间，并跨会话保持上下文。

<table>
  <tr>
    <th>类别</th>
    <th>XMem (Gemini 3-flash)</th>
    <th>Backboard.io (GPT-4o)</th>
    <th>Mastra (GPT-4o)</th>
    <th>Supermemory (GPT-4o)</th>
  </tr>
  <tr><td><strong>多会话</strong></td><td><strong>93.6</strong></td><td>91.7</td><td>79.7</td><td>71.43</td></tr>
  <tr><td><strong>时间推理</strong></td><td><strong>94.5</strong></td><td>91.7</td><td>85.7</td><td>76.69</td></tr>
  <tr><td><strong>单会话助手</strong></td><td><strong>96.43</strong></td><td>98.2</td><td>82.1</td><td>96.43</td></tr>
  <tr><td><strong>单会话用户</strong></td><td><strong>97.1</strong></td><td>97.1</td><td>98.6</td><td>97.14</td></tr>
  <tr><td><strong>知识更新</strong></td><td><strong>91.2</strong></td><td>93.6</td><td>85.9</td><td>88.46</td></tr>
  <tr><td><strong>单会话偏好</strong></td><td><strong>87.0</strong></td><td>90.0</td><td>73.3</td><td>70.0</td></tr>
</table>

> XMem 在所有类别上都接近 Backboard.io，在会话召回和偏好跟踪上接近满分。总体而言，XMem 比 Mastra 高 **9.2 分**，比 Supermemory 高 **11.8 分**。

### 我们如何做基准测试
- **评估**：使用 Gemini 作为 LLM-as-Judge，并采用结构化评分规则
- **公平性**：所有系统都使用完全相同的对话历史和查询进行测试

## 快速开始

### 本地 XMem

```bash
npx create-xmem@latest
cd xmem
npm run dev
```

适用于 Windows、macOS 和 Linux。它会创建本地 XMem 工作区、安装后端、启动本地存储、构建 Chrome 扩展，并在 `http://localhost:8000` 启动 API。

本地前置条件：

- Git
- Node.js 20+
- Python 3.11+
- Docker Desktop
- Ollama，除非你在 `.env` 中添加云端 LLM key

设置完成后，从以下路径加载扩展：

```text
repos/xmem-extension/dist
```

Chrome 路径：`chrome://extensions` -> 启用开发者模式 -> 加载已解压的扩展程序。

### 本地命令

```bash
npm run setup
npm run start
npm run verify
npm run doctor
```

如果 `.env` 包含真实的云端 LLM key，XMem 会使用该提供商，并使用 FastEmbed 保持嵌入在本地运行。如果没有配置云端 key，XMem 会回退到本地 Ollama，并在设置过程中拉取所需的本地模型。

### 上下文可移植性

```bash
npm run context:export
npm run context:import -- --file ./exports/xmem-context.json
npm run context:sync -- --file ./exports/xmem-context.json --server https://api.xmem.in --api-key <key>
```

`context:export` 会写出一个本地上下文包，之后可以重新导入或同步到 XMem 服务器。

### 索引仓库

```bash
python -m src.scanner.runner \
  --org your-org \
  --repo your-repo \
  --url https://github.com/your-org/your-repo.git \
  --enrich
```

> [!TIP]
> 对于完全本地、无云依赖的设置：
> ```ini
> FALLBACK_ORDER='["ollama"]'
> EMBEDDING_PROVIDER=ollama
> VECTOR_STORE_PROVIDER=pgvector
> ```
> 然后安装本地扩展依赖：`pip install -e ".[local]"`

## 配置

XMem 可高度配置。你可以覆盖任意 Agent 的模型、调整兜底链，或调节质量与速度之间的取舍。

<table>
  <tr>
    <th>设置</th>
    <th>默认值</th>
    <th>描述</th>
  </tr>
  <tr><td><code>FALLBACK_ORDER</code></td><td><code>openrouter,gemini,claude,openai</code></td><td>提供商故障切换顺序</td></tr>
  <tr><td><code>DEEPSEEK_API_KEY</code></td><td>empty</td><td>用于官方 OpenAI 兼容端点的 DeepSeek API key</td></tr>
  <tr><td><code>MIMO_API_KEY</code></td><td>empty</td><td>用于官方 OpenAI 兼容端点的小米 MiMo API key</td></tr>
  <tr><td><code>CLASSIFIER_MODEL</code></td><td>default model</td><td>覆盖 classifier agent 的模型</td></tr>
  <tr><td><code>JUDGE_MODEL</code></td><td>default model</td><td>覆盖 judge agent 的模型</td></tr>
  <tr><td><code>RETRIEVAL_MODEL</code></td><td>default model</td><td>覆盖检索综合模型</td></tr>
  <tr><td><code>EMBEDDING_MODEL</code></td><td><code>gemini-embedding-001</code></td><td>文本嵌入模型</td></tr>
  <tr><td><code>EMBEDDING_PROVIDER</code></td><td><code>auto</code></td><td>auto, gemini, bedrock, ollama, fastembed</td></tr>
  <tr><td><code>VECTOR_STORE_PROVIDER</code></td><td><code>pinecone</code></td><td>pinecone, pgvector, chroma, sqlite</td></tr>
  <tr><td><code>PINECONE_DIMENSION</code></td><td><code>768</code></td><td>嵌入向量维度</td></tr>
  <tr><td><code>RATE_LIMIT</code></td><td><code>60</code></td><td>每分钟 API 请求数</td></tr>
  <tr><td><code>TEMPERATURE</code></td><td><code>0.4</code></td><td>LLM 生成温度</td></tr>
</table>
