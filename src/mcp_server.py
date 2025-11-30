import os
import sys
import json
from fastmcp import FastMCP

# Add src to path to import tools
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools import list_tables_in_directory, get_schema, query_parquet_files

# Initialize FastMCP
mcp = FastMCP("RCA Data MCP Server")

@mcp.tool()
def search(query: str) -> str:
    """
    Search for data tables by keyword, OR execute a SQL query (SELECT ...) to analyze data.
    """
    query_lower = query.lower().strip()
    
    # SQL Execution Mode
    if "select" in query_lower and "from" in query_lower:
        print(f"[MCP] Executing SQL: {query}")
        try:
            # Get all parquet files to register as views
            tables_json = list_tables_in_directory("data")
            tables = json.loads(tables_json)
            
            all_files = []
            if isinstance(tables, list):
                all_files = [f"data/{t['filename']}" for t in tables]
            
            # Execute query
            result = query_parquet_files(all_files, query)
            
            # Format result as a list of dictionaries for the 'results' key
            return json.dumps({
                "results": [{
                    "id": "sql_result",
                    "title": "SQL Query Result",
                    "text": str(result),
                    "url": "sql://query"
                }]
            })
        except Exception as e:
            return json.dumps({"results": [], "error": str(e)})

    # Keyword Search Mode
    print(f"[MCP] Searching for: {query}")
    try:
        tables_json = list_tables_in_directory("data")
        tables = json.loads(tables_json)
        
        results = []
        if isinstance(tables, list):
            for table in tables:
                if not query or query == "all" or query in table['filename'].lower():
                    # Ensure each result has id, title, text, and url
                    results.append({
                        "id": table['filename'],
                        "title": table['filename'],
                        "text": f"Table: {table['filename']}", # Added text field
                        "url": f"file://data/{table['filename']}",
                    })
        
        return json.dumps({"results": results})
    except Exception as e:
        return json.dumps({"results": [], "error": str(e)})

@mcp.tool()
def fetch(id: str) -> str:
    """
    Fetch schema and sample data for a specific file/table ID.
    """
    print(f"[MCP] Fetching: {id}")
    try:
        # Check if id is a path or just filename
        file_path = id
        if not id.startswith("data/") and not os.path.exists(id):
             if os.path.exists(f"data/{id}"):
                 file_path = f"data/{id}"
        
        schema_json = get_schema(file_path)
        
        # Use query tool to get sample data
        sample_query = f"SELECT * FROM '{file_path}' LIMIT 10"
        sample_data = query_parquet_files([file_path], sample_query)
        
        full_text = f"Schema:\n{schema_json}\n\nSample Data (First 10 rows):\n{sample_data}"
        
        # Return the result object directly, not wrapped in another object
        # The fetch tool should return the document object itself
        result_object = {
            "id": id,
            "title": id,
            "text": full_text,
            "url": f"file://{file_path}",
            "metadata": {"source": "local_parquet"}
        }
        return json.dumps(result_object)
    except Exception as e:
        return json.dumps({"error": str(e)})

if __name__ == "__main__":
    # Force SSE transport on port 8000
    mcp.run(transport="sse", port=8000)
