# AGENTS.md

给编码助手/自动化 Agent 的项目阅读约定。目标：用尽量少的上下文建立正确认知，避免整份加载长文档。

## 当前基线（2026-10-24，先看这里）

| 项目 | 当前值 |
| --- | --- |
| 默认 rerank | `qwen3-rerank`（`config/rag.yml` 是事实源） |
| 其他模型 | 对话 `qwen3.5-flash`；Embedding `text-embedding-v4`；Answerability `qwen-flash` |
| 索引 | `agent_v2`：239 子块 / 96 父块；`agent_dd`：1543 子块 / 450 父块（道德经实验集合） |
| 评测 | Golden Set 71 条 + 道德经 32 条；`eval/results/baseline.json` 是 gte 时代历史基线 |
| 测试 | 37 条 stdlib unittest |
| 模型对比 | `docs/rag-improvement/07-rerank模型对比报告.md`（结论有边界，不要当普适定律） |

> 文档里的指标是**快照**：换索引、换模型、换阈值后不可直接比较；以代码、`config/*.yml` 和最新评测报告为准。

## 阅读顺序

1. `docs/PROJECT_OVERVIEW.md`：先建立全局（定位、分层、主流程、关键约束）。
2. `README.md` 的「快速开始」「配置入口速查」：只在需要命令/环境变量时读。
3. `docs/ARCHITECTURE.md` 的对应章节：按 `PROJECT_OVERVIEW.md` 第 8 节索引按需读，不要默认整份加载。
4. 改 RAG 前先读 `docs/rag-improvement/00-进度备忘录.md`，再按阶段读 `01`~`05`；`06` 是长任务工作方法，`07` 是 rerank 模型对比与选型。
5. 源码：需要精确行为时再读；代码是最终事实源。
6. `CHANGELOG.md`：只在需要知道“最近改了什么 / 为什么改”时读；它是 Keep a Changelog 格式的人工摘要，不是 git log。

## 不要做的事

- 不要为了建立认知去整份读 `docs/*.svg`。图是给人看的，文字模块表更适合 AI。但**核对/更新图里的文字时**，用 grep 只取 `<text>` 节点定位，改完用 XML 解析校验一次。
- 不要假设 `utils/runtime.py` 存在；当前运行时初始化由 `env_patch/` 的 `.pth + sitecustomize.py` 完成。
- 不要把 `ARCHITECTURE.md` 里的“示例/预留”当成已接入主链路；状态标记以 ✅ / 🧪 / ⚙️ / ⚠️ 为准。
- 不要假设 rerank 模型名或 Answerability 阈值；先读 `config/rag.yml`。
- 不要把不同 rerank 模型的分数阈值混用；`qwen3-vl-rerank` 的分数在 precision≥0.9 下没有可行拒答阈值。
- 不要把 `agent_dd`（道德经实验集合）当主知识库；主链路默认是 `agent_v2`。
- 默认 `rag.vector_store` 会按 `config/chroma.yml` 的 `exclude_keywords` 跳过道德经；要建实验库，用显式的 `--collection agent_dd --include-keywords 道德经` 命令。
- 不要在未固定“索引快照 + Golden Set + rerank 模型 + Answerability 模式”时比较指标。

## 开始任务前

1. 读 `PROJECT_OVERVIEW.md` + `ARCHITECTURE.md` 目标章节，不要整份加载。
2. 建 TODO 与进度备忘（参考 `docs/rag-improvement/00-进度备忘录.md`）：目标、非目标、验收、终止、回滚点。
3. 跑基线：`uv run -m unittest discover -s tests -v`；涉及检索时先确认 collection / rerank model / answerability mode。
4. 如果任务涉及效果，先建/确认 Golden Set 和 baseline，再改系统。

## 修改后的文档动作

按 `docs/ARCHITECTURE.md` 第 16.3 节同步：

- 新增/删除工具 → `README.md` 能力表、`ARCHITECTURE.md` 5.2/13.1、`architecture.svg` 工具层。
- 新增/删除中间件 → `ARCHITECTURE.md` 5.2/8/11、`architecture.svg` 编排层。
- 改 RAG → `ARCHITECTURE.md` 7、`rag-pipeline.svg`、`docs/rag-improvement/` 对应阶段文档。
- 改 rerank 模型/阈值 → `ARCHITECTURE.md` 3/7.2/9.2/12、`README.md` 技术栈/已知限制、`PROJECT_OVERVIEW.md` 2/5、`rag-pipeline.svg`；重跑评测，更新 `07` 或新增报告。
- 改评测/Golden Set → `docs/rag-improvement/01`、`07`、`README.md` 4.7；更新 `eval/results/baseline.json` 前确认没有大面积降级。
- 改配置 → `ARCHITECTURE.md` 9、`README.md` 配置速查。
- 改模型/依赖/目录 → 对应技术栈、拓扑与目录树章节；**图也要同步**：`architecture.svg` 的模型/存储层节点与副标题（改动落在 RAG 链上的模型时，再同步 `rag-pipeline.svg` 的 ChatModel 节点）。
- 任何值得记录的改动（新增/变更/废弃/移除/修复）→ 归入 `CHANGELOG.md` 的 `## [Unreleased]`，按 Keep a Changelog 分组（Added / Changed / Deprecated / Removed / Fixed / Security）写英文标题 + 中文描述；发布时才改成版本标题。不要把增量日志写进架构正文。

## 收尾前

1. `uv run -m unittest discover -s tests -v` 通过；涉及检索时跑一次 smoke eval。
2. 更新 TODO/进度备忘：完成项、已知缺口、下一步。
3. 同步 README / ARCHITECTURE / PROJECT_OVERVIEW / SVG / CHANGELOG。
4. `git status` 检查：临时文件、运行产物、敏感信息、是否误覆盖 baseline。

## 冲突处理

文档与代码冲突时，以代码为准；修正文档，并在 `docs/ARCHITECTURE.md` 第 12 节更新技术债状态。历史变更交给 Git / `CHANGELOG.md`，不要把增量日志写进架构正文。