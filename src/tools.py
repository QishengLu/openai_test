"""
tools.py — RCA parquet 数据工具

与 Deep_Research/src/rca_tools.py 实现完全对齐（去掉 LangChain @tool 装饰器）。
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Union, List

TOKEN_LIMIT = 5000


def _import_duckdb():
    try:
        import duckdb
        return duckdb
    except ImportError:
        raise ImportError("duckdb is required. Install it with: pip install duckdb")


def _serialize_datetime(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: _serialize_datetime(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_serialize_datetime(item) for item in obj]
    else:
        return obj


def _estimate_token_count(text: str) -> int:
    average_chars_per_token = 3
    return (len(text) + average_chars_per_token - 1) // average_chars_per_token


def _enforce_token_limit(payload: str, context: str) -> str:
    token_estimate = _estimate_token_count(payload)
    if token_estimate <= TOKEN_LIMIT:
        return payload

    current_size = len(json.loads(payload)) if payload.startswith("[") else None
    suggested_limit = None
    if current_size:
        ratio = TOKEN_LIMIT / token_estimate
        suggested_limit = max(1, int(current_size * ratio * 0.8))

    suggestion_parts = [
        "The query result is too large. Please adjust your query:",
        "  • Reduce the LIMIT value" + (f" (try LIMIT {suggested_limit})" if suggested_limit else ""),
        "  • Filter rows with WHERE clauses to reduce result size",
        "  • Select only necessary columns instead of SELECT *",
        "  • Use aggregation (COUNT, SUM, AVG) instead of retrieving raw rows",
    ]

    warning = {
        "error": "Result exceeds token budget",
        "context": context,
        "estimated_tokens": token_estimate,
        "token_limit": TOKEN_LIMIT,
        "rows_returned": current_size,
        "suggested_limit": suggested_limit,
        "suggestion": "\n".join(suggestion_parts),
    }
    return json.dumps(warning, ensure_ascii=False, indent=2)


ALLOWED_STEMS = {
    "normal_logs", "abnormal_logs",
    "normal_traces", "abnormal_traces",
    "normal_metrics", "abnormal_metrics",
    "normal_metrics_histogram", "abnormal_metrics_histogram",
    "normal_metrics_sum", "abnormal_metrics_sum",
}


def _sanitize_column_name(name: str) -> str:
    """Replace dots in column names with underscores to avoid DuckDB dot-notation ambiguity."""
    return name.replace(".", "_")


def _build_rename_select(parquet_path: str) -> str:
    """Build a SELECT clause that renames dot-containing columns for a parquet file."""
    duckdb = _import_duckdb()
    conn = duckdb.connect(":memory:")
    try:
        result = conn.execute(f"SELECT * FROM read_parquet('{parquet_path}') LIMIT 0")
        columns = [desc[0] for desc in result.description]
    finally:
        conn.close()

    needs_rename = any("." in col for col in columns)
    if not needs_rename:
        return "*"

    parts = []
    for col in columns:
        if "." in col:
            parts.append(f'"{col}" AS {_sanitize_column_name(col)}')
        else:
            parts.append(col)
    return ", ".join(parts)


def _validate_parquet_files(parquet_files: Union[str, List[str]]) -> List[str]:
    if isinstance(parquet_files, str):
        parquet_files = [parquet_files]
    for file_path in parquet_files:
        if not Path(file_path).exists():
            raise FileNotFoundError(
                f"Parquet file not found: {file_path}\n"
                f"Please check the file path and ensure the file exists. "
                f"You may use 'list_tables_in_directory' to discover available parquet files."
            )
    return parquet_files


def list_tables_in_directory(directory: str) -> str:
    """
    List all parquet files in a directory with metadata.

    Args:
        directory: Directory path to search for parquet files

    Returns:
        JSON string containing list of files with metadata
    """
    duckdb = _import_duckdb()

    dir_path = Path(directory)
    if not dir_path.exists():
        return json.dumps({"error": f"Directory not found: {directory}"})
    if not dir_path.is_dir():
        return json.dumps({"error": f"Path is not a directory: {directory}"})

    files_info = []
    cwd = Path.cwd()

    for file_path in sorted(dir_path.rglob("*.parquet")):
        if file_path.stem not in ALLOWED_STEMS:
            continue
        file_path_str = str(file_path)
        if file_path.is_absolute():
            try:
                file_path_str = str(file_path.relative_to(cwd))
            except ValueError:
                file_path_str = str(file_path)

        try:
            conn = duckdb.connect(":memory:")
            row_count_result = conn.execute(
                f"SELECT COUNT(*) FROM read_parquet('{file_path}')"
            ).fetchone()
            row_count = row_count_result[0] if row_count_result else 0
            result = conn.execute(f"SELECT * FROM read_parquet('{file_path}') LIMIT 0")
            column_count = len(result.description)
            conn.close()
            files_info.append({
                "filename": file_path.name,
                "path": str(file_path),
                "row_count": row_count,
                "column_count": column_count,
            })
        except Exception as e:
            files_info.append({"filename": file_path.name, "path": str(file_path), "error": str(e)})

    result_json = json.dumps(files_info, ensure_ascii=False, indent=2)
    return _enforce_token_limit(result_json, "list_tables_in_directory")


def _get_schema_one(parquet_file: str) -> dict:
    duckdb = _import_duckdb()

    if not Path(parquet_file).exists():
        return {"error": f"Parquet file not found: {parquet_file}"}

    conn = duckdb.connect(":memory:")
    try:
        result = conn.execute(f"SELECT * FROM read_parquet('{parquet_file}') LIMIT 0")
        schema = [
            {"name": _sanitize_column_name(desc[0]), "type": str(desc[1])}
            for desc in result.description
        ]
        row_count_result = conn.execute(
            f"SELECT COUNT(*) FROM read_parquet('{parquet_file}')"
        ).fetchone()
        row_count = row_count_result[0] if row_count_result else 0
        return {"file": parquet_file, "row_count": row_count, "columns": schema}
    except Exception as e:
        return {"error": f"Failed to extract schema: {str(e)}"}
    finally:
        conn.close()


def get_schema(parquet_files: Union[str, List[str]]) -> str:
    """
    Get schema information of a parquet file, or a list of parquet files.

    Args:
        parquet_files: Path to a parquet file, or list of paths for batch lookup

    Returns:
        JSON string containing file metadata — single object if one file, list if multiple
    """
    if isinstance(parquet_files, str):
        result_json = json.dumps(_get_schema_one(parquet_files), ensure_ascii=False, indent=2)
    else:
        result_json = json.dumps(
            [_get_schema_one(f) for f in parquet_files], ensure_ascii=False, indent=2
        )
    return _enforce_token_limit(result_json, "get_schema")


def query_parquet_files(parquet_files: Union[str, List[str]], query: str, limit: int = 10) -> str:
    """
    Query parquet files using SQL syntax for data analysis and exploration.

    Args:
        parquet_files: Path(s) to parquet file(s)
        query: SQL query to execute
        limit: Maximum number of records to return

    Returns:
        JSON string of query results
    """
    duckdb = _import_duckdb()
    try:
        parquet_files = _validate_parquet_files(parquet_files)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)})

    conn = duckdb.connect(":memory:")
    table_names: set = set()

    try:
        cwd = Path.cwd()
        relative_files = []
        for file_path in parquet_files:
            fp = Path(file_path)
            if fp.is_absolute():
                try:
                    file_path = str(fp.relative_to(cwd))
                except ValueError:
                    file_path = str(fp)
            relative_files.append(file_path)
        parquet_files = relative_files

        for file_path in parquet_files:
            base_name = Path(file_path).stem
            table_name = base_name
            counter = 1
            while table_name in table_names:
                table_name = f"{base_name}_{counter}"
                counter += 1
            table_names.add(table_name)
            select_clause = _build_rename_select(file_path)
            conn.execute(
                f"CREATE VIEW {table_name} AS SELECT {select_clause} FROM read_parquet('{file_path}')"
            )

        result = conn.execute(query).fetchall()
        columns = [desc[0] for desc in conn.description]
        rows = _serialize_datetime([dict(zip(columns, row)) for row in result])

        if len(rows) > limit:
            rows = rows[:limit]

        result_json = json.dumps(rows, ensure_ascii=False, indent=2)
        return _enforce_token_limit(result_json, "query_parquet_files")

    except Exception as e:
        error_msg = str(e)
        if "syntax error" in error_msg.lower() or "parser error" in error_msg.lower():
            return json.dumps({
                "error": f"SQL syntax error in query: {error_msg}",
                "query": query,
                "available_tables": list(table_names),
            })
        elif "catalog" in error_msg.lower() or "table" in error_msg.lower():
            return json.dumps({
                "error": f"Table reference error: {error_msg}",
                "query": query,
                "available_tables": list(table_names),
            })
        else:
            return json.dumps({
                "error": f"Query execution failed: {error_msg}",
                "query": query,
                "available_tables": list(table_names),
            })
    finally:
        conn.close()
