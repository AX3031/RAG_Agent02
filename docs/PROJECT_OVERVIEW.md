# 项目速览（1 页版）

> 用途：给人和 AI 的快速上下文。命令与环境见 [README](../README.md)；深入设计见 [ARCHITECTURE.md](ARCHITECTURE.md)。
>
> 代码基线：`dev01` 工作区快照（2026-10-23）。
>
> 配套图：[architecture.svg](architecture.svg)（分层架构）、[rag-pipeline.svg](rag-pipeline.svg)（RAG 数据流）；AI 阅读约定见 [AGENTS.md](../AGENTS.md)。

**本文目录**：[1 一句话](#1-一句话) · [2 技术栈](#2-技术栈) · [3 分层结构](#3-分层结构) · [4 请求流程](#4-一次请求怎么走) · [5 RAG/报告](#5-rag-与报告场景) · [6 目录](#6-目录速查) · [7 约束](#7-关键约束) · [8 问题索引](#8-按问题找位置) · [9 AI 约定](#9-给-ai-的阅读约定)

## 1. 一句话

基于 **LangChain 1.4 `create_agent`（LangGraph）** 的机器人智能客服原型：Streamlit UI + 工具调用 + Chroma RAG + DashScope 模型，支持报告场景动态提示词、PII 脱敏、模型降级，以及短期/本地 JSON/PostgreSQL 三种会话记忆。RAG 已改造为**检索子 Agent**：结构化 `evidence[] + answerable`、Dense+BM25+RRF+Rerank、父子分块、Golden Set 离线评测、可回答性判定与 trace。

## 2. 技术栈

| 组件 | 版本 | 作用 |
| --- | --- | --- |
| langchain / langgraph | 1.4.0 / 1.2.11 | Agent 编排与图执行 |
| langgraph-checkpoint-postgres / psycopg | 3.2.0 / 3.3.6 | 会话记忆：内存（默认）/ 本地 JSON / PostgreSQL |
| langchain-openai | 1.6.6 | `init_chat_model` 构造的 `ChatOpenAI` 调用 DashScope 兼容端点 |
| langchain-community | 0.4.2 | `DashScopeEmbeddings`、PDF/TXT Loader |
| langchain-chroma / chromadb | 1.1.0 / 1.5.9 | 本地向量库 |
| dashscope | 1.27.5 | 通义千问 / Embedding / gte-rerank SDK |
| streamlit | 1.64.0 | 对话 UI 与流式渲染 |

模型：`qwen3.5-flash`（对话）、`text-embedding-v4`（向量化）、`qwen-flash`（Answerability 判定）、`gte-rerank-v2`（精排）。

评测：Golden Set 71 条（论语 50 + 本地需求文档 21），stdlib 实现 Recall@10 / MRR@10 / NDCG@10。

## 3. 分层结构

| 层 | 关键组件 | 职责 |
| --- | --- | --- |
| 表现层 | `app.py` | 收集输入、展示流式输出、保存 UI 会话状态 |
| 编排层 | `ReactAgent` + 6 个中间件 | 组装模型/工具/提示词，驱动 ReAct 循环 |
| 工具层 | 9 个 `@tool` | 证据检索、摘要、天气、位置、用户、月份、外部 CSV、场景开关、百度联网搜索 |
| 服务层 | `RetrievalService`、`RagSummarizeService`、`VectorStoreService` | 混合检索/证据组装、受控生成、Chroma 入库与索引运维 |
| 模型/存储 | `ChatOpenAI`、`DashScopeEmbeddings`、Chroma、索引清单/父块、记忆后端 | 推理、向量化、上下文持久化 |
| 横切 | `config/`、`prompts/`、`utils/`、`eval/`、`logs/` | 配置、提示词、路径、日志、评测、环境补丁 |

## 4. 一次请求怎么走

1. `app.py` 接收输入，调用 `ReactAgent.execute_stream(query, thread_id)`。
2. `create_agent` 构建 LangGraph，传入 `context={"report": False}`、`config={"thread_id": ...}`。
3. 中间件在模型调用外层做模型降级与 PII 脱敏，并在调用前记录日志、选择系统提示词。
4. 模型返回 `tool_calls`，工具节点执行工具；结果以 `ToolMessage` 回到模型。
5. 循环直到模型返回最终 `AIMessage`。
6. `stream_mode="values"` 每个状态快照被 `yield`，UI 用 `write_stream` 展示，最后 `st.rerun()`。

## 5. RAG 与报告场景

**RAG 入库**：`data/*.txt|pdf` → `TextLoader`/`PyPDFLoader` → 结构化**父子分块**（子块 240 字 / overlap 40；父块 900 字）→ 每个块写确定性 `chunk_id` 与 `doc_id/version/parent_id/page/title_path` → `DashScopeEmbeddings` → `Chroma`（collection `agent_v2`）。`storage/index_manifest.json` 记录文件→版本→块清单；文件变化时先写新版本、再删旧块；磁盘删除可 prune。旧 `md5.text` 不再是权威清单。

**RAG 检索**：主 Agent 调用 `retrieve_knowledge(query)` 或 `rag_summarize(query)` → Dense（Chroma，k=15）+ Sparse（自研 BM25，k=15）→ RRF(k=60) → `gte-rerank-v2` 精排 → 按 `parent_id` 父子扩展 → 证据组装（总数/长度上限）→ Answerability 判定（qwen-flash + 阈值兜底 + 熔断）→ 返回 `evidence[] + answerable + status + trace`。纯检索 P95 约 300ms；含 rerank + Answerability 的平均约 700ms。

**报告场景**：模型调用 `fill_context_for_report` → `monitor_tool` 设置 `runtime.context["report"]=True` → 下一次模型调用由 `report_prompt_switch` 切换到 `report_prompt.txt`。标记只在本次请求内有效。

## 6. 目录速查

```text
agent/        Agent 组装、9 个工具、6 个中间件
rag/          检索服务、契约、父子分块、索引清单、BM25、可回答性、telemetry
eval/         Golden Set、检索指标、三模式评测、阈值校准
model/        Chat / Embedding 工厂
utils/        配置、路径、文件、日志、提示词加载
config/       rag / chroma / agent / prompts
prompts/      main / rag_summarize / report
data/         知识文件 + external/records.csv（本地需求文档转换的 txt 不入库）
chroma_db/    向量库（运行时生成，不入库）
storage/      索引清单与父块（运行时生成，不入库）
memorys/      记忆方式工厂 + 短期/JSON/PostgreSQL checkpointer
docs/         架构文档、SVG 与 rag-improvement 分阶段讲解
env_patch/    .pth + sitecustomize.py
tests/        35 条 stdlib unittest
```

## 7. 关键约束

- **默认短期记忆**：输入框左侧的记忆按钮可选短期（默认，不存档）/ 本地 JSON 存档 / PostgreSQL；侧边栏管理会话 ID；短期记忆刷新页面或重启即丢失。
- **消息块级流式**：`stream_mode="values"` 只过滤输出 `AIMessage` 文本，工具中间消息不再显示；UI 只保存最后一个 chunk。
- **提示词已补全**：三套提示词已补全角色、工具规则、输出结构与边界；新增 `retrieve_knowledge` 的“证据不足不得硬生成”硬约束。
- **外部 CSV 为空**：`data/external/records.csv` 当前 0 字节；`fetch_external_data` 注解为 `str`、实际返回 `dict`。
- **知识库更新/删除已传播**：同 MD5 跳过；内容变化先写新版本再删旧块；`--rebuild` / `--delete-doc` / prune 可用。
- **检索已混合**：Dense+BM25+RRF+rerank；rerank 额度耗尽时熔断降级为 RRF（`status=degraded`）。
- **可回答性**：qwen-flash 判定 + 阈值兜底；证据不足时主 Agent 强制输出拒答文本。
- **未实现**：鉴权、限流、多租户、CI、文件上传 UI、软删除/GC、tokenizer 分块；PII 仅覆盖模型输入/输出，日志未系统性脱敏。

## 8. 按问题找位置

| 我想知道 | 看这里 |
| --- | --- |
| 项目怎么跑起来？ | [README 快速开始](../README.md#4-快速开始) |
| 整体分几层、用了什么？ | [ARCHITECTURE 第 4 节](ARCHITECTURE.md#4-总体架构) + [architecture.svg](architecture.svg) |
| 一次请求怎么走？ | [ARCHITECTURE 第 6 节](ARCHITECTURE.md#6-运行时主链路) |
| RAG 怎么入库/检索？ | [ARCHITECTURE 第 7 节](ARCHITECTURE.md#7-rag-子系统) + [rag-pipeline.svg](rag-pipeline.svg) |
| RAG 怎么评测/调参？ | [docs/rag-improvement/01-阶段0-评估标尺.md](rag-improvement/01-阶段0-评估标尺.md) |
| 证据契约/索引更新？ | [docs/rag-improvement/02-阶段1-证据契约与索引平面.md](rag-improvement/02-阶段1-证据契约与索引平面.md) |
| 长任务怎么做、怎么复盘？ | [docs/rag-improvement/06-长期改造经验与方法论.md](rag-improvement/06-长期改造经验与方法论.md) |
| 报告场景怎么切换？ | [ARCHITECTURE 第 8 节](ARCHITECTURE.md#8-报告场景与动态提示词) |
| 配置在哪改？ | [ARCHITECTURE 第 9 节](ARCHITECTURE.md#9-配置体系) + [README 配置速查](../README.md#7-配置入口速查) |
| 想加工具/上传/记忆？ | [ARCHITECTURE 第 13 节](ARCHITECTURE.md#13-扩展指南) |
| 有什么坑/技术债？ | [ARCHITECTURE 第 12 节](ARCHITECTURE.md#12-已知问题与技术债) |
| 启动失败/日志在哪？ | [ARCHITECTURE 第 11 节](ARCHITECTURE.md#11-可观测性与错误处理) + [16.2 常用命令](ARCHITECTURE.md#162-常用命令) |
| 代码文件对应关系？ | [ARCHITECTURE 16.1](ARCHITECTURE.md#161-关键文件索引) |
| 最近改了什么、为什么改？ | [CHANGELOG](../CHANGELOG.md) |

## 9. 给 AI 的阅读约定

完整约定见 [AGENTS.md](../AGENTS.md)。核心四点：

- 先读本文件，再按第 8 节索引只读 `ARCHITECTURE.md` 对应章节，不要整份加载。
- 不要为了建立认知去整份读 `docs/*.svg`；图是给人看的，文字模块表更适合 AI（改图时只 grep `<text>` 节点定位）。
- 代码是最终事实源；文档与代码冲突时，以代码为准并更新文档。
- 增量变更写 [CHANGELOG.md](../CHANGELOG.md)，不要塞进架构正文。