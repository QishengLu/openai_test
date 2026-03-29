# openai_test — 改造说明

## 背景

本项目原本使用基于文本指令解析（`<<SEARCH:>>` / `<<FETCH:>>`）的提示词注入方式模拟工具调用。
现已重构为与 **thinkdepthai（Deep_Research）rollout 结构完全对齐**的标准 RCA Agent，接入 RolloutRunner 评测框架。

---

## 当前架构

### 入口文件

| 文件 | 用途 |
|------|------|
| `agent_runner.py` | **主入口**，RolloutRunner 调用（stdin/stdout 接口） |
| `test_local.py` | 本地冒烟测试，无需 RolloutRunner |
| `src/tools.py` | DuckDB parquet 工具实现（与 Deep_Research/src/rca_tools.py 对齐） |

### 数据流

```
RolloutRunner (run_rollout.py)
  │  加载 rca.yaml → format() 填充占位符
  │  构造 payload JSON → stdin
  ▼
agent_runner.py main()
  │  读 stdin JSON（6 字段）
  │  strip_think_tool(system_prompt / compress_up)  ← 去除 think_tool 引用
  │  user_prompt += data_dir 数据路径提示
  ▼
run_research_loop()                         ← LLM + tool 循环
  │  messages = [system, user]
  │  for _ in range(50):
  │      LLM call (tools=TOOLS, tool_choice="auto")
  │      if no tool_calls: break
  │      execute each tool → append tool messages
  ▼
run_compress()                              ← 压缩为 CausalGraph JSON
  │  messages = [compress_sp] + trajectory + [compress_up]
  │  LLM call (no tools)
  ▼
stdout: {"output": "<CausalGraph JSON>", "trajectory": [...]}
```

---

## 与 thinkdepthai 的对比

| 维度 | openai_test (o3-deep-research) | thinkdepthai (kimi-k2) |
|------|-------------------------------|------------------------|
| **模型** | openai/o3-deep-research (OpenRouter) | openai:kimi-k2-0905-preview |
| **框架** | 纯 Python while 循环 + OpenAI SDK | LangGraph StateGraph + LangChain |
| **Tools** | 3 个（无 think_tool） | 4 个（含 think_tool） |
| **Tool 注册** | OpenAI JSON schema 列表 | `model.bind_tools(RCA_TOOLS)` |
| **消息格式** | 原生 OpenAI dict，无需转换 | LangChain Message 对象 → convert_trajectory() |
| **Prompt** | rca.yaml（去除 think_tool 引用） | rca.yaml + rca_think_prompt 追加 |
| **迭代上限** | 50 次 LLM call | recursion_limit=100（图节点执行次数） |
| **stdin/stdout** | ✅ 完全一致 | ✅ |
| **Tool 实现** | src/tools.py（对齐 rca_tools.py） | src/rca_tools.py |

---

## Tools（与 thinkdepthai 对齐，去掉 think_tool）

| Tool | 功能 | 实现位置 |
|------|------|---------|
| `list_tables_in_directory` | 递归列出 parquet 文件（ALLOWED_STEMS 过滤） | `src/tools.py` |
| `get_schema` | 批量获取 parquet schema，自动重命名点列名 | `src/tools.py` |
| `query_parquet_files` | DuckDB SQL 查询，TOKEN_LIMIT=5000，limit 参数 | `src/tools.py` |

---

## Prompt 处理

Prompt 由 RolloutRunner 从 `rca.yaml` 加载并格式化后通过 stdin 传入。
`agent_runner.py` 对收到的 prompt 做以下处理后直接使用：

1. `system_prompt`：去除 `4. **think_tool**...` 整行
2. `compress_up`：去除 `**Exclude**: think_tool calls...` 整行

---

## RolloutRunner 配置

```yaml
# RolloutRunner/configs/agents/o3_deep_research.yaml
name: o3_deep_research
cmd: ["python", "agent_runner.py"]
cwd: /home/nn/SOTA-agents/test/openai_test
exp_id: rollout_o3_deep_research
model_name: openai/o3-deep-research
agent_type: o3_deep_research
concurrency: 1
timeout: 1800
data_dir: /home/nn/SOTA-agents/RolloutRunner/data
```

---

## 本地测试

```bash
cd /home/nn/SOTA-agents/test/openai_test

# 使用本地 data/ 目录的 parquet 文件做冒烟测试
python test_local.py

# 自定义 data_dir
python test_local.py --data_dir /path/to/incident/data

# 手动构造 payload 测试 agent_runner.py
echo '{"question":"...","system_prompt":"...","user_prompt":"...","compress_system_prompt":"...","compress_user_prompt":"...","data_dir":"data"}' | python agent_runner.py
```

---

## 环境变量（.env）

```
DEEPRESEARCH_API_KEY=sk-or-v1-...   # OpenRouter API Key
DEEPRESEARCH_API_URL=https://openrouter.ai/api/v1
DEEPRESEARCH_MODEL=openai/o3-deep-research
```
