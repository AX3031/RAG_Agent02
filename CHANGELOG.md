# Changelog

本文件记录 project2 的所有重要变更。

格式遵循 [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)（分组标题保留英文，便于工具解析），版本号遵循 [Semantic Versioning](https://semver.org/spec/v2.0.0.html)。仓库当前没有 tag，未发布的改动累积在 `## [Unreleased]`；发布时把该节整体挪到新的版本标题下（例如 `## [0.1.0] - 2026-10-23`），并在文末补 compare 链接。

## [Unreleased]

### Added

- RAG 检索子 Agent：`rag/retrieval_service.py` —— Dense(Chroma) + Sparse(自研 BM25) → RRF(k=60) → `gte-rerank-v2` 精排 → 父子扩展 → 证据组装 → Answerability。
- `rag/contract.py`：`evidence[] / answerable / status / degraded_stages / latency_ms / index_version` 契约与 5 态状态机；`validate_evidence_ids` 引用校验。
- `rag/chunking.py`：结构化父子分块与确定性 `doc_id/version/chunk_id/parent_id`；子块 240 / overlap 40，父块 900。
- `rag/index_store.py`：`IndexManifest`（文件→版本→块清单）与 `ParentStore`（父块），临时文件 + `os.replace` 原子写。
- `rag/sparse_index.py`：中文 unigram + bigram 的 BM25；`allowed_ids` 召回前过滤；进程内缓存。
- `rag/answerability.py`：qwen-flash YES/NO 判定 + 快速路径 + 阈值兜底 + 熔断。
- `rag/telemetry.py`：`Trace`（request_id/trace_id/分段延迟/脱敏日志）与 TTL+LRU 缓存。
- `eval/`：Golden Set（论语 50 + 本地需求文档 21）、Recall@K / HitRate@K / MRR@K / NDCG@K、随机基线、三模式评测、阈值校准、`results/baseline.json`。
- `tests/`：35 条 stdlib unittest（指标、BM25、分块、契约、清单、RRF/缓存、Agent 拒答护栏）。
- `agent/tools/agent_tools.py`：`retrieve_knowledge(query)` 工具，返回结构化证据 JSON。
- `agent/react_agent.py`：`_knowledge_insufficient` 确定性护栏——证据不足时强制输出「根据现有资料无法回答该问题」。
- `docs/rag-improvement/`：进度备忘录 + 阶段 0-4 的入门讲解与反思；`06-长期改造经验与方法论.md`（长任务工作方法：开始、阶段门、复盘、效率与避免幻觉）。
- `data/论语.pdf` 测试语料；`eval/prepare_local_corpus.py` 生成不纳入版本管理的需求文档语料。
- `config/rag.yml` 新增 `rerank_model_name` 与 `retrieval:` 段（候选数、RRF、精排、Answerability、熔断、缓存）。
- `config/chroma.yml` 新增 `parent_chunk_size`、`index_manifest`、`parent_store`、`scope_tags`。
- `memorys/memory_factory.py`：`create_checkpointer(memory_mode)` 统一入口，`DEFAULT_MEMORY_MODE = "short_term"`、`MEMORY_MODE_LABELS`（界面文案 ↔ 模式值）。
- `memorys/file_checkpointer.py`：`FileCheckpointSaver` —— 继承 LangGraph `InMemorySaver`，把 `storage/writes/blobs` 序列化为 JSON 并原子写入 `memorys/chat_history/agent_checkpoints.json`（`bytes ↔ base64`、`tuple ↔ 标签结构`、非字符串 dict 键 ↔ 键值对列表；临时文件 + `os.replace`；`threading.RLock`；进程内单例 `get_file_checkpointer()`）。
- `agent/react_agent.py`：`ReactAgent(memory_mode=...)`、`clear_memory(thread_id)`、`load_ui_messages(thread_id)`；`_default_memory_mode()` 兼容读取 `USE_POSTGRES_CHECKPOINTER`。
- `app.py`：对话输入框左侧的记忆按钮（`st.bottom` + 水平容器 + `st.popover` + `st.radio`），点击弹出三种记忆方式列表；侧边栏「会话管理」= 会话 ID、新建会话、清空记忆。
- `CHANGELOG.md`。

### Changed

- RAG 输出从「一段不可追溯的文本」改为结构化证据契约；`rag_summarize` 保留为受控摘要，新增 `retrieve_knowledge`。
- `rag/vector_store.py`：入库改为父子分块 + 稳定 chunk_id + 清单登记；支持增量更新、`--rebuild`、`--delete-doc`、prune；不再使用 `md5.text` 作为权威去重依据。
- Chroma 集合由 `agent` 改为 `agent_v2`，与旧脏集合隔离；切分参数由 200/20 改为子块 240/40、父块 900。
- `prompts/rag_summarize.txt`：证据分隔符、evidence_id 引用、非法引用约束。
- `prompts/main_prompt.txt`：新增 `retrieve_knowledge`、证据不足硬约束、引用出处要求。
- `README.md`、`docs/PROJECT_OVERVIEW.md`、`docs/ARCHITECTURE.md`、`docs/architecture.svg`、`docs/rag-pipeline.svg` 同步 RAG 改造。
- `.gitignore` 增加 `storage/`、`chroma_db/`、`logs/`、`md5.text`、本地评测集与语料。
- 解除运行产物的历史跟踪（技术债 TD-16）：`chroma_db/`、`logs/`、`md5.text`、`memorys/chat_history/`、各模块 `__pycache__/*.pyc` 共 29 个文件移出 Git 索引，本地文件保留；`.gitignore` 规则自此生效，`docs/ARCHITECTURE.md` 10.2 记下做法与沙箱拦截 `rm` 的踩坑。
- `docs/rag-improvement/06-长期改造经验与方法论.md` 已同步进索引：`AGENTS.md` 阅读顺序、`README.md` 文档地图、`docs/PROJECT_OVERVIEW.md` 问题索引。
- 三种会话记忆统一走 LangGraph checkpointer（`short_term` → `InMemorySaver`；`file` → `FileCheckpointSaver`；`postgres` → 原有 `PostgresSaver`）；`execute_stream` 每次只传当前一条 `HumanMessage`，历史由 `thread_id` 自动恢复。
- `memorys/postgres_checkpointer.py`：连接池 `timeout` 由 30 秒改为 5 秒，数据库未启动时快速失败并由界面回退为短期记忆，不再卡住页面。
- `memorys/temporary_memory.py`、`memorys/persistent_memory.py` 保留为示例、不接入主链路：其接口是 `BaseChatMessageHistory`，缺少 `get_tuple/put/put_writes/list/delete_thread`，且 `InMemoryChatMessageHistory` 在 LangChain 1.6.4 已标记废弃。
- `config/rag.yml`：`backup_chat_model_name` 由 `deepseek-v4-flash-0731` 改为 `glm-5`。
- 记忆方式的界面文案（`memorys/memory_factory.py`、`app.py`）：选项改为「短期记忆（默认，不存档记忆）/ 本地记忆（以JSON格式本地存档）/ PostgreSQL（存档入数据库，需配置）」；短期记忆的提示补上“刷新页面”也会丢（Streamlit 刷新会重建 `st.session_state` 与 Agent）；弹层 caption 不再重复显示会话 ID。
- 文档同步（记忆改造）：`README.md`（能力表、4.5 启动、配置速查、目录树与文档索引）、`docs/PROJECT_OVERVIEW.md`（分层、关键约束、问题索引）、`docs/ARCHITECTURE.md`（5.2、5.3、6.4、9.2、9.6、12、13.4、16.1、16.3）、`docs/architecture.svg`（记忆方式说明、文件型会话历史状态）。
- `.gitignore`：忽略运行时生成的 `memorys/chat_history/`。

### Removed

- `rag/vector_store.py` 中的 `check_md5_hex` / `save_md5_hex` 逻辑被 `IndexManifest` 取代（`md5.text` 文件保留但不再读写）。
- `agent/react_agent.py` 的 `_build_checkpointer()`：checkpointer 的创建改由 `memorys/memory_factory.py` 负责。

### Fixed

- TD-07「知识库只增不删」：内容变化先写新版本再删旧块；实测 239→238→239（删除后重建）。
- 证据不足时主 Agent 使用自有知识硬编：增加代码级拒答护栏（不依赖提示词遵守）。
- 生成答案引用越界：`validate_evidence_ids` 校验 `【ev_003】` / `【参考资料3】` 必须来自本次检索。
- rerank 额度耗尽（HTTP 403）导致请求失败：403 不重试，连续 3 次失败熔断 120s，降级为 RRF 并标记 `degraded_stages`。
- 旧 `md5.text` 并发/非原子问题：新清单使用临时文件 + `os.replace`；旧的 `md5.text` 仅保留为历史文件。
- 聊天界面不再回显用户问题与工具中间结果，只输出不带 `tool_calls` 的最终 `AIMessage` 文本（技术债 TD-03）。
- 修复回答为空时 `response_messages[-1]` 抛出的 `IndexError`（技术债 TD-04）。

### Security

- 检索证据按不可信数据处理：固定分隔符 `<<<EVIDENCE ... EVIDENCE>>>`，提示词声明不执行证据内指令。
- RAG trace 只记录 `query_hash`、`doc_id`、`evidence_id`，不记录完整 query 与证据正文。
- `scope` 过滤在 Dense/BM25 召回前生效（接口形态；真实鉴权尚未接入）。
