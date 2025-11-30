import json
import os
import sys
import time
import requests
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Add src to path to import tools
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools import list_tables_in_directory, get_schema, query_parquet_files
from prompt.system_prompt import get_system_prompt

class RCAAgent:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.history: List[Dict[str, str]] = []

    def call_llm_api(self, prompt: str) -> str:
        """
        Call OpenRouter API for Deep Research with MCP.
        """
        print("\n[System] Calling OpenRouter API (openai/o3-deep-research)...")
        
        api_key = os.getenv("DEEPRESEARCH_API_KEY")
        base_url = os.getenv("DEEPRESEARCH_API_URL", "https://openrouter.ai/api/v1")
        model_name = os.getenv("DEEPRESEARCH_MODEL", "openai/o3-deep-research")
        mcp_server_url = os.getenv("MCP_SERVER_URL")
        
        if not api_key:
            raise ValueError("Please set DEEPRESEARCH_API_KEY in .env")
        if not mcp_server_url:
            raise ValueError("Please set MCP_SERVER_URL in .env")
            
        # Remove trailing slash
        mcp_server_url = mcp_server_url.rstrip("/")
            
        client = OpenAI(
            api_key=api_key, 
            base_url=base_url,
            timeout=3600
        )
        
        try:
            # Use chat.completions.create as requested for OpenRouter
            # We pass the MCP tool configuration in 'extra_body'
            
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                extra_headers={
                    "HTTP-Referer": "https://github.com/SOTA-agents/api_test",
                    "X-Title": "RCA Agent",
                },
                extra_body={
                    "tools": [
                        {
                            "type": "mcp",
                            "server_label": "rca_data_server",
                            "server_url": mcp_server_url,
                            "allowed_tools": ["search", "fetch"],
                            "require_approval": "never"
                        }
                    ]
                }
            )
            return completion.choices[0].message.content
        except Exception as e:
            raise Exception(f"API Call failed: {e}")

    def save_history(self, output_path: str = "experiments/openai/output.json"):
        """Save the conversation history to a JSON file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
            print(f"\n[System] History saved to {output_path}")
        except Exception as e:
            print(f"\n[System] Error saving history: {e}")

    def run(self):
        print("Starting RCA Agent with Deep Research (MCP Mode)...")
        print("IMPORTANT: Ensure your MCP server is running and exposed via a public URL.")
        print("Set MCP_SERVER_URL in your .env file.")
        
        # Initial System Prompt
        system_prompt = get_system_prompt()
        
        full_input = f"""
{system_prompt}

Note on Data Access:
You have access to a local data server via the 'mcp' tool.
- To search for tables, use the 'search' tool with keywords.
- To get table schema and sample data, use the 'fetch' tool with the table filename.
- **To execute SQL queries**, use the 'search' tool with a valid SQL SELECT statement (e.g., "SELECT * FROM 'table.parquet' LIMIT 5"). The server will detect the SQL syntax and execute it.

Please conduct a deep research analysis on the above problem using the available data tools.
"""
        self.history.append({"role": "user", "content": full_input})
        
        try:
            # Call Deep Research (single step)
            response = self.call_llm_api(full_input)
            
            print("\nAnalysis Complete.")
            print(response)
            
            self.history.append({"role": "assistant", "content": response})
            
        except Exception as e:
            print(f"Error: {e}")
        
        # Save history at the end of the run
        self.save_history()

if __name__ == "__main__":
    agent = RCAAgent()
    agent.run()
