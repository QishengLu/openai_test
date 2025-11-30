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
    def __init__(self, data_dir: str = "data", output_path: str = "experiments/openai/output.json"):
        self.data_dir = data_dir
        self.output_path = output_path
        self.history: List[Dict[str, str]] = []

    def call_llm_api(self, prompt: str) -> str:
        """
        Call OpenRouter API for Deep Research with Prompt-based Tool Execution.
        This bypasses the API limitation where o3-deep-research doesn't support standard 'function' tools.
        """
        print("\n[System] Calling OpenRouter API (openai/o3-deep-research)...")
        
        api_key = os.getenv("DEEPRESEARCH_API_KEY")
        base_url = os.getenv("DEEPRESEARCH_API_URL", "https://openrouter.ai/api/v1")
        model_name = os.getenv("DEEPRESEARCH_MODEL", "openai/o3-deep-research")
        
        if not api_key:
            raise ValueError("Please set DEEPRESEARCH_API_KEY in .env")
            
        client = OpenAI(
            api_key=api_key, 
            base_url=base_url,
            timeout=3600
        )
        
        # Append tool instructions to the prompt
        tool_instructions = """
\n[TOOL INSTRUCTIONS]
You have access to a local data server. To use it, you MUST output a specific command format on a new line.
DO NOT output markdown code blocks for these commands. Just the command text.

Available Commands:
1. SEARCH: `<<SEARCH: your search query>>`
   - Use this to find tables or execute SQL.
   - Example: <<SEARCH: sales data>>
   - Example: <<SEARCH: SELECT * FROM 'data/sales.parquet' LIMIT 5>>

2. FETCH: `<<FETCH: file_id>>`
   - Use this to get the schema and sample data of a file.
   - Example: <<FETCH: data/sales.parquet>>

When you need data, output ONLY the command and stop generating. I will execute it and give you the result.
After you get the result, continue your analysis.
"""
        
        # Check if this is the first turn (simple check)
        # Type hint: List[Dict[str, str]] is compatible with OpenAI's expected message format
        # but Pylance might complain if it expects specific TypedDicts.
        # We can cast it or ignore the type error as the runtime behavior is correct.
        current_messages: List[Dict[str, Any]] = [
            {"role": "user", "content": prompt + tool_instructions}
        ]
        
        # We need a loop to handle multiple tool turns
        max_turns = 5
        current_turn = 0
        
        while current_turn < max_turns:
            current_turn += 1
            
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=current_messages, # type: ignore
                    max_tokens=20000,
                    # No tools parameter to avoid 400 errors
                    extra_headers={
                        "HTTP-Referer": "https://github.com/SOTA-agents/api_test",
                        "X-Title": "RCA Agent",
                    }
                )
                
                content = completion.choices[0].message.content
                
                # Capture reasoning if available
                reasoning = getattr(completion.choices[0].message, 'reasoning', None)
                if reasoning:
                    print(f"\n[Agent] Model Reasoning:\n{reasoning[:200]}...")
                    # Save reasoning to history for debugging/analysis
                    self.history.append({
                        "role": "assistant", 
                        "content": f"[REASONING]\n{reasoning}\n\n[CONTENT]\n{content or ''}"
                    })

                # Handle case where content is empty but reasoning is present (o3-deep-research behavior)
                if not content:
                    if reasoning:
                        # If the model reasoned but didn't output content, it might have hit the token limit
                        # or it's still thinking. We can try to extract intent from reasoning if possible,
                        # but usually we need the model to output the command in 'content'.
                        
                        # Check if the reasoning contains the command (sometimes models leak it there)
                        if "<<SEARCH:" in reasoning:
                            content = reasoning # Treat reasoning as content for command extraction
                            print("[System] Found command in reasoning block, proceeding...")
                        elif "<<FETCH:" in reasoning:
                            content = reasoning
                            print("[System] Found command in reasoning block, proceeding...")
                        else:
                            # If no command in reasoning, and finish_reason is 'length', we need to increase max_tokens
                            if completion.choices[0].finish_reason == 'length':
                                raise ValueError("Model ran out of tokens while reasoning. Try increasing max_tokens.")
                            
                            raise ValueError("Model returned empty content (but provided reasoning). It may not have followed the instruction to output a command.")
                    else:
                        print(f"[Debug] Full Completion Object: {completion}")
                        raise ValueError("Model returned empty content")
                
                print(f"\n[Agent] Model Response:\n{content[:200]}...")
                
                # Check for tool commands
                tool_result = None
                
                if "<<SEARCH:" in content:
                    start = content.find("<<SEARCH:") + 9
                    end = content.find(">>", start)
                    if end != -1:
                        query = content[start:end].strip()
                        print(f"\n[Agent] Detected SEARCH command: {query}")
                        
                        # Execute Search
                        try:
                            query_lower = query.lower()
                            if "select" in query_lower and "from" in query_lower:
                                tables_json = list_tables_in_directory("data")
                                tables = json.loads(tables_json)
                                all_files = [f"data/{t['filename']}" for t in tables] if isinstance(tables, list) else []
                                result = query_parquet_files(all_files, query)
                                tool_result = f"SQL Result:\n{result}"
                            else:
                                tables_json = list_tables_in_directory("data")
                                tool_result = f"Search Results:\n{tables_json}"
                        except Exception as e:
                            tool_result = f"Search Error: {e}"

                elif "<<FETCH:" in content:
                    start = content.find("<<FETCH:") + 8
                    end = content.find(">>", start)
                    if end != -1:
                        file_id = content[start:end].strip()
                        print(f"\n[Agent] Detected FETCH command: {file_id}")
                        
                        # Execute Fetch
                        try:
                            file_path = file_id
                            if not file_id.startswith("data/") and not os.path.exists(file_id):
                                if os.path.exists(f"data/{file_id}"):
                                    file_path = f"data/{file_id}"
                            
                            schema_json = get_schema(file_path)
                            sample_query = f"SELECT * FROM '{file_path}' LIMIT 5"
                            sample_data = query_parquet_files([file_path], sample_query)
                            tool_result = f"File: {file_id}\nSchema:\n{schema_json}\n\nSample Data:\n{sample_data}"
                        except Exception as e:
                            tool_result = f"Fetch Error: {e}"
                
                # If a tool was executed, add result and continue loop
                if tool_result:
                    print(f"[System] Tool Output Length: {len(tool_result)} chars")
                    
                    # Add assistant's command message
                    current_messages.append({"role": "assistant", "content": content})
                    
                    # Add tool result as user message (since we are simulating tools)
                    current_messages.append({
                        "role": "user", 
                        "content": f"[TOOL RESULT]\n{tool_result}\n\nPlease continue your analysis based on this data."
                    })
                    continue # Loop again to get model's next response
                
                # If no tool command found, this is the final answer
                return content

            except Exception as e:
                raise Exception(f"API Call failed: {e}")
                
        return "Analysis stopped after maximum turns."

    def save_history(self):
        """Save the conversation history to a JSON file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
            print(f"\n[System] History saved to {self.output_path}")
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
    import argparse
    parser = argparse.ArgumentParser(description="Run RCA Agent")
    parser.add_argument("--data_dir", default="data", help="Directory containing data files")
    parser.add_argument("--output_path", default="experiments/openai/output.json", help="Path to save output JSON")
    args = parser.parse_args()

    agent = RCAAgent(data_dir=args.data_dir, output_path=args.output_path)
    agent.run()
