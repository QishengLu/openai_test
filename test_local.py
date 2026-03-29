#!/usr/bin/env python
"""
本地冒烟测试脚本 — 复现 RolloutRunner 的 build_payload 逻辑，
直接调用 agent_runner.py，无需启动 RolloutRunner 和数据库。

用法:
  python test_local.py
  python test_local.py --data_dir data --question "自定义问题描述"
"""
import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import yaml


PROMPTS_PATH = Path(__file__).parent.parent.parent / "SOTA-agents" / "RolloutRunner" / "configs" / "prompts" / "rca.yaml"
# fallback: 相对于本文件
if not PROMPTS_PATH.exists():
    PROMPTS_PATH = Path(__file__).parent.parent.parent / "RolloutRunner" / "configs" / "prompts" / "rca.yaml"
if not PROMPTS_PATH.exists():
    # 再试一次绝对路径
    PROMPTS_PATH = Path("/home/nn/SOTA-agents/RolloutRunner/configs/prompts/rca.yaml")


DEFAULT_QUESTION = """\
You are investigating a system failure in namespace `ts0`.
Abnormal Period (Fault Injection): 2025-07-23 14:10:23 to 2025-07-23 14:14:23 UTC
Normal Period (Baseline): 2025-07-23 14:06:23 to 2025-07-23 14:10:23 UTC
Identify the root cause service and fault propagation path.
"""


def load_prompts() -> dict:
    if not PROMPTS_PATH.exists():
        raise FileNotFoundError(f"rca.yaml not found at {PROMPTS_PATH}")
    with open(PROMPTS_PATH) as f:
        return yaml.safe_load(f)


def build_payload(question: str, data_dir: str) -> dict:
    prompts = load_prompts()
    today = date.today().isoformat()
    return {
        "question": question,
        "system_prompt": prompts["RCA_ANALYSIS_SP"].format(date=today),
        "user_prompt": prompts["RCA_ANALYSIS_UP"].format(incident_description=question),
        "compress_system_prompt": prompts["COMPRESS_FINDINGS_SP"].format(date=today),
        "compress_user_prompt": prompts["COMPRESS_FINDINGS_UP"].format(
            date=today, incident_description=question
        ),
        "data_dir": data_dir,
    }


def main():
    parser = argparse.ArgumentParser(description="Local smoke test for agent_runner.py")
    parser.add_argument(
        "--data_dir",
        default=str(Path(__file__).parent / "data"),
        help="Path to directory containing parquet files",
    )
    parser.add_argument(
        "--question",
        default=DEFAULT_QUESTION,
        help="Incident description / question",
    )
    args = parser.parse_args()

    payload = build_payload(args.question, args.data_dir)

    print(f"[test_local] data_dir = {args.data_dir}")
    print(f"[test_local] model    = openai/o3-deep-research (from .env)")
    print("[test_local] Sending payload to agent_runner.py ...\n")

    agent_script = Path(__file__).parent / "agent_runner.py"
    proc = subprocess.run(
        [sys.executable, str(agent_script)],
        input=json.dumps(payload),
        capture_output=False,
        text=True,
        cwd=str(Path(__file__).parent),
    )
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
