#!/usr/bin/env python
"""
agent_runner_deep_research.py — OpenAI Deep Research (o3/o4-mini) RCA 测评接口

Deep Research 模型不支持 function calling，采用 Prompt-based Tool Execution：
  1. 在 prompt 中注入 <<SEARCH:...>> / <<FETCH:...>> 指令格式
  2. 解析模型输出中的指令，本地执行 parquet 查询
  3. 将结果追加到对话中，循环直到模型给出最终分析

stdin:  JSON { question, system_prompt, user_prompt,
               compress_system_prompt, compress_user_prompt, data_dir }
stdout: JSON { output (CausalGraph JSON), trajectory (OpenAI role 格式), usage }
"""
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, "/home/nn/SOTA-agents/RolloutRunner")
from src.usage_tracker import UsageTracker

_tracker = UsageTracker()
_tracker.install_openai_hooks()

# 清理 RolloutRunner 路径和 src 模块缓存，避免与本项目的 src 包冲突
sys.path.remove("/home/nn/SOTA-agents/RolloutRunner")
for _mod in list(sys.modules):
    if _mod == "src" or _mod.startswith("src."):
        del sys.modules[_mod]


import yaml
from datetime import date

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent / ".env")

sys.path.insert(0, str(Path(__file__).parent / "src"))
from tools import list_tables_in_directory, get_schema, query_parquet_files

# ── 加载 Deep Research 专用 prompt ────────────────────────────────────────────

PROMPTS_PATH = Path("/home/nn/SOTA-agents/RolloutRunner/configs/prompts/rca_deep_research.yaml")


def load_deep_research_prompts() -> dict[str, str]:
    """加载 Deep Research 专用 prompt 原始模板。"""
    with open(PROMPTS_PATH) as f:
        return yaml.safe_load(f)


# ── Tool instructions injected into prompt ────────────────────────────────────

TOOL_INSTRUCTIONS = """

## Data Query Tools

You have access to local telemetry data. To query it, output a command on a NEW LINE using the exact format below. After outputting a command, STOP generating and wait for the result.

### Available Commands:

1. **SEARCH** — List tables or execute SQL:
   `<<SEARCH: your query>>`
   - Keyword search: `<<SEARCH: abnormal>>`
   - SQL query: `<<SEARCH: SELECT * FROM abnormal_metrics LIMIT 5>>`

2. **FETCH** — Get schema + sample data for a table:
   `<<FETCH: filename.parquet>>`

### Rules:
- Output ONE command per turn, then STOP.
- After receiving [TOOL RESULT], continue your analysis.
- When analysis is complete, output your final answer WITHOUT any commands.
- You may issue multiple rounds of commands across turns.
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def strip_think_tool(prompt: str) -> str:
    """Remove all think_tool references from a prompt string."""
    prompt = re.sub(
        r"\n?[^\n]*4\.\s+\*\*think_tool\*\*[^\n]*",
        "",
        prompt,
    )
    prompt = re.sub(
        r"\n?[^\n]*\*\*Exclude\*\*: think_tool calls[^\n]*",
        "",
        prompt,
    )
    return prompt


def strip_markdown_json(text: str) -> str:
    """Strip ```json ... ``` wrapper from LLM output."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def execute_search(query: str, data_dir: str) -> str:
    """Execute a SEARCH command: SQL query or keyword search."""
    query_lower = query.lower().strip()
    if "select" in query_lower and "from" in query_lower:
        try:
            tables_json = list_tables_in_directory(data_dir)
            tables = json.loads(tables_json)
            all_files = (
                [os.path.join(data_dir, t["filename"]) for t in tables]
                if isinstance(tables, list)
                else []
            )
            result = query_parquet_files(all_files, query)
            return f"SQL Result:\n{result}"
        except Exception as e:
            return f"Search Error: {e}"
    else:
        try:
            tables_json = list_tables_in_directory(data_dir)
            return f"Available Tables:\n{tables_json}"
        except Exception as e:
            return f"Search Error: {e}"


def execute_fetch(file_id: str, data_dir: str) -> str:
    """Execute a FETCH command: get schema + sample data."""
    try:
        file_path = file_id
        if not file_id.startswith(data_dir) and not os.path.exists(file_id):
            potential = os.path.join(data_dir, file_id)
            if os.path.exists(potential):
                file_path = potential
        schema_json = get_schema(file_path)
        sample_query = f"SELECT * LIMIT 5"
        sample_data = query_parquet_files([file_path], sample_query)
        return f"File: {file_id}\nSchema:\n{schema_json}\n\nSample Data:\n{sample_data}"
    except Exception as e:
        return f"Fetch Error: {e}"


def parse_tool_command(content: str) -> tuple[str | None, str | None]:
    """Parse <<SEARCH: ...>> or <<FETCH: ...>> from model output.
    Returns (command_type, argument) or (None, None)."""
    for pattern, cmd_type in [
        (r"<<SEARCH:\s*(.+?)>>", "SEARCH"),
        (r"<<FETCH:\s*(.+?)>>", "FETCH"),
    ]:
        m = re.search(pattern, content, re.DOTALL)
        if m:
            return cmd_type, m.group(1).strip()
    return None, None


# ── Core loop ─────────────────────────────────────────────────────────────────

def run_research_loop(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    data_dir: str,
    max_iterations: int = 100,
) -> list[dict[str, Any]]:
    """
    Prompt-based tool execution loop.
    Returns the full message list (for trajectory output).
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for i in range(max_iterations):
        print(f"[DeepResearch] Turn {i + 1}/{max_iterations}", file=sys.stderr)
        response = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            extra_headers={
                "HTTP-Referer": "https://github.com/SOTA-agents/rca-eval",
                "X-Title": "RCA Agent",
            },
        )
        msg = response.choices[0].message
        content = msg.content or ""

        # 有些 deep research 模型会把内容放在 reasoning 里
        if not content:
            reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
            if reasoning:
                for tag in ("<<SEARCH:", "<<FETCH:"):
                    if tag in reasoning:
                        content = reasoning
                        print("[DeepResearch] Found command in reasoning block", file=sys.stderr)
                        break
            if not content:
                # 真的没有内容，记录空消息后退出
                messages.append({"role": "assistant", "content": ""})
                break

        messages.append({"role": "assistant", "content": content})

        # 解析工具指令
        cmd_type, cmd_arg = parse_tool_command(content)
        if cmd_type is None:
            # 没有工具指令 → 最终回答
            break

        # 执行工具
        if cmd_type == "SEARCH":
            tool_result = execute_search(cmd_arg, data_dir)
        else:
            tool_result = execute_fetch(cmd_arg, data_dir)

        print(f"[DeepResearch] {cmd_type}: {cmd_arg[:80]}... → {len(tool_result)} chars", file=sys.stderr)

        # 模拟 tool 调用：将 assistant 的工具消息转换为带 tool_calls 的格式
        # 同时用 user message 传回结果（deep research 不支持 tool role）
        tool_call_id = f"call_prompt_{i}"
        messages[-1]["tool_calls"] = [
            {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": cmd_type.lower(),
                    "arguments": json.dumps({"query" if cmd_type == "SEARCH" else "id": cmd_arg}),
                },
            }
        ]
        # 用 user role 传回结果（因为 deep research 不接受 tool role）
        messages.append({
            "role": "user",
            "content": f"[TOOL RESULT]\n{tool_result}\n\nPlease continue your analysis based on this data. If you need more data, use another command. Otherwise, provide your final analysis.",
        })

    return messages


def run_compress(
    client: OpenAI,
    model: str,
    compress_sp: str,
    compress_up: str,
    trajectory: list[dict[str, Any]],
) -> str:
    """Compress research trajectory into CausalGraph JSON."""
    # 过滤掉 tool_calls 字段（compress 阶段不需要）
    clean_trajectory = []
    for m in trajectory:
        entry = {"role": m["role"], "content": m["content"]}
        clean_trajectory.append(entry)

    messages = (
        [{"role": "system", "content": compress_sp}]
        + clean_trajectory
        + [{"role": "user", "content": compress_up}]
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore[arg-type]
        extra_headers={
            "HTTP-Referer": "https://github.com/SOTA-agents/rca-eval",
            "X-Title": "RCA Agent",
        },
    )
    return response.choices[0].message.content or ""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    payload = json.loads(sys.stdin.read())
    data_dir = payload.get("data_dir", "")

    # 从 RolloutRunner 传入的 payload 中提取 incident_description
    # （user_prompt 里包含了 incident_description，直接用 question 字段更简洁）
    incident_description = payload.get("question", payload.get("user_prompt", ""))

    # 加载 Deep Research 专用 prompt（替换 RolloutRunner 传入的通用 prompt）
    prompts = load_deep_research_prompts()
    today = date.today().isoformat()
    system_prompt = prompts["RCA_ANALYSIS_SP"].format(date=today)
    user_prompt = prompts["RCA_ANALYSIS_UP"].format(
        date=today, incident_description=incident_description,
    )
    compress_sp = prompts["COMPRESS_FINDINGS_SP"].format(date=today)
    compress_up = prompts["COMPRESS_FINDINGS_UP"].format(
        date=today, incident_description=incident_description,
    )

    if data_dir:
        user_prompt = (
            f"{user_prompt}\n\n## Data Location\n\n"
            f"The telemetry data for this incident is located at: `{data_dir}`\n\n"
            f"**Your first action MUST be**: `<<SEARCH: all>>`"
        )

    api_key = os.getenv("DEEPRESEARCH_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("DEEPRESEARCH_API_URL", "https://openrouter.ai/api/v1")
    model = os.getenv("DEEPRESEARCH_MODEL", "openai/o4-mini-deep-research")

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=3600)

    trajectory = run_research_loop(client, model, system_prompt, user_prompt, data_dir)

    compressed = run_compress(client, model, compress_sp, compress_up, trajectory)

    # 构建输出 trajectory（排除 system，转换为标准 OpenAI role 格式）
    output_trajectory = [m for m in trajectory if m["role"] != "system"]

    result = {
        "output": strip_markdown_json(compressed),
        "trajectory": output_trajectory,
        "usage": _tracker.get_usage(),
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
