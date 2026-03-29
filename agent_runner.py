#!/usr/bin/env python
"""
agent_runner.py — OpenAI o3-deep-research RCA 测评接口

stdin:  JSON { question, system_prompt, user_prompt,
               compress_system_prompt, compress_user_prompt, data_dir }
stdout: JSON { output (CausalGraph JSON), trajectory (OpenAI 格式) }
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


from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent / ".env")

sys.path.insert(0, str(Path(__file__).parent / "src"))
from tools import list_tables_in_directory, get_schema, query_parquet_files


# ── Tool definitions (OpenAI function calling format) ────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_tables_in_directory",
            "description": (
                "List all parquet files in the specified directory with metadata "
                "(row counts, column counts). Call this FIRST to discover available data files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory path containing parquet files",
                    }
                },
                "required": ["directory"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_schema",
            "description": (
                "Get column schema for one or multiple parquet files. "
                "Pass a single path string or a list of paths. "
                "Call with all 10 file paths at once (batch mode) to understand every column before querying."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "parquet_files": {
                        "description": "Single parquet file path or list of paths",
                        "oneOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}},
                        ],
                    }
                },
                "required": ["parquet_files"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_parquet_files",
            "description": (
                "Query parquet files using SQL syntax. "
                "The file stem is the table name (e.g., 'abnormal_metrics', 'abnormal_traces'). "
                "Always use LIMIT to control result size."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "parquet_files": {
                        "description": "Single parquet file path or list of paths to register as SQL tables",
                        "oneOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}},
                        ],
                    },
                    "query": {
                        "type": "string",
                        "description": "SQL SELECT query to execute against the registered tables",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of records to return (default 10)",
                    },
                },
                "required": ["parquet_files", "query"],
            },
        },
    },
]

TOOLS_BY_NAME = {
    "list_tables_in_directory": list_tables_in_directory,
    "get_schema": get_schema,
    "query_parquet_files": query_parquet_files,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def execute_tool(name: str, args: dict) -> str:
    fn = TOOLS_BY_NAME.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        return fn(**args)
    except Exception as e:
        return json.dumps({"error": str(e)})


def strip_think_tool(prompt: str) -> str:
    """Remove all think_tool references from a prompt string."""
    # 1. Numbered bullet: "4. **think_tool**: ..." (in RCA_ANALYSIS_SP Available Tools)
    prompt = re.sub(
        r"\n?[^\n]*4\.\s+\*\*think_tool\*\*[^\n]*",
        "",
        prompt,
    )
    # 2. Exclude bullet: "- **Exclude**: think_tool calls ..." (in COMPRESS_FINDINGS_UP)
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


# ── Core loop ─────────────────────────────────────────────────────────────────

def run_research_loop(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_iterations: int = 50,
) -> list[dict[str, Any]]:
    """
    LLM → tool_node loop until no more tool calls.
    Returns the full message list (system excluded from output).
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            tools=TOOLS,        # type: ignore[arg-type]
            tool_choice="auto",
        )
        msg = response.choices[0].message

        assistant_entry: dict[str, Any] = {
            "role": "assistant",
            "content": msg.content or "",
        }
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_entry)

        if not msg.tool_calls:
            break

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result = execute_tool(tc.function.name, args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )

    return messages


def run_compress(
    client: OpenAI,
    model: str,
    compress_sp: str,
    compress_up: str,
    trajectory: list[dict[str, Any]],
) -> str:
    """Compress research trajectory into CausalGraph JSON."""
    messages = (
        [{"role": "system", "content": compress_sp}]
        + trajectory
        + [{"role": "user", "content": compress_up}]
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore[arg-type]
    )
    return response.choices[0].message.content or ""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    payload = json.loads(sys.stdin.read())

    system_prompt = payload["system_prompt"]
    user_prompt = payload["user_prompt"]
    compress_sp = payload["compress_system_prompt"]
    compress_up = payload["compress_user_prompt"]
    data_dir = payload.get("data_dir", "")

    # Remove think_tool references from all prompts (tool not available for this agent)
    system_prompt = strip_think_tool(system_prompt)
    compress_up = strip_think_tool(compress_up)

    if data_dir:
        user_prompt = (
            f"{user_prompt}\n\n## Data Location\n\n"
            f"The telemetry data for this incident is located at: `{data_dir}`\n\n"
            f'Start by calling `list_tables_in_directory(directory="{data_dir}")` '
            f"to discover available parquet files."
        )

    api_key = os.getenv("DEEPRESEARCH_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("DEEPRESEARCH_API_URL", "https://api.shubiaobiao.cn/v1")
    model = os.getenv("DEEPRESEARCH_MODEL", "openai/o3-deep-research")

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=3600)

    trajectory = run_research_loop(client, model, system_prompt, user_prompt)

    compressed = run_compress(client, model, compress_sp, compress_up, trajectory)

    # Exclude system message from trajectory output
    output_trajectory = [m for m in trajectory if m["role"] != "system"]

    result = {
        "output": strip_markdown_json(compressed),
        "trajectory": output_trajectory,
        "usage": _tracker.get_usage(),
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
