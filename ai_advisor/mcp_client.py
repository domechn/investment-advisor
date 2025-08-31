from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, List
import logging


class MCPNotAvailable(RuntimeError):
    pass


def _expand_env(obj: Any) -> Any:
    """Recursively expand environment variables in strings in a JSON-like structure."""
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, list):
        return [_expand_env(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    return obj


def _load_mcp_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    path = Path(config_path or "mcp/mcp.json")
    if not path.exists():
        raise MCPNotAvailable(f"MCP config not found at {path}. Create it based on mcp/mcp.json.example.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return _expand_env(data)
    except Exception as e:
        raise MCPNotAvailable(f"Failed to read MCP config at {path}: {e}")

def get_mcp_client(config_path: Optional[str] = None):
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient  # type: ignore
    except Exception as e:
        raise MCPNotAvailable(
            "langchain-mcp-adapters not installed. `pip install langchain-mcp-adapters` or run without --use-mcp."
        ) from e
    cfg = _load_mcp_config(config_path)
    return MultiServerMCPClient(cfg)


def wrap_mcp_tools_for_logging(tools: List[Any], logger: Optional[logging.Logger] = None) -> List[Any]:
    """Monkeypatch invoke/ainvoke on original LangChain tools to add logging.

    Preserves the original BaseTool types so create_react_agent accepts them.
    Safe to call multiple times; skips tools already patched.
    """
    log = logger or logging.getLogger("mcp.tools")

    def _truncate(obj: Any, max_len: int = 500) -> str:
        try:
            s = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
        except Exception:
            s = str(obj)
        if len(s) > max_len:
            return s[: max_len - 3] + "..."
        return s

    for t in tools:
        try:
            # Avoid double patch
            if getattr(t, "__mcp_logged__", False):
                continue
            name = getattr(t, "name", t.__class__.__name__)
            # Patch sync invoke
            if hasattr(t, "invoke") and callable(getattr(t, "invoke")):
                _orig_invoke = getattr(t, "invoke")

                def _patched_invoke(self, input: Any, config: Optional[Dict[str, Any]] = None, __orig=_orig_invoke, __name=name):
                    try:
                        log.info("MCP tool call: %s input=%s", __name, _truncate(input))
                    except Exception:
                        pass
                    out = None
                    try:
                        out = __orig(input, config=config)
                        return out
                    finally:
                        try:
                            log.info("MCP tool result: %s output=%s", __name, _truncate(out))
                        except Exception:
                            pass

                # bind method to instance
                import types
                t.invoke = types.MethodType(_patched_invoke, t)

            # Patch async ainvoke
            if hasattr(t, "ainvoke") and callable(getattr(t, "ainvoke")):
                _orig_ainvoke = getattr(t, "ainvoke")

                async def _patched_ainvoke(self, input: Any, config: Optional[Dict[str, Any]] = None, __orig=_orig_ainvoke, __name=name):
                    try:
                        log.info("MCP tool call: %s input=%s", __name, _truncate(input))
                    except Exception:
                        pass
                    out = None
                    try:
                        out = await __orig(input, config=config)
                        return out
                    finally:
                        try:
                            log.info("MCP tool result: %s output=%s", __name, _truncate(out))
                        except Exception:
                            pass

                import types
                t.ainvoke = types.MethodType(_patched_ainvoke, t)

            setattr(t, "__mcp_logged__", True)
        except Exception:
            # Best-effort: ignore patch failures for a tool
            continue

    return tools


__all__ = [
    "MCPNotAvailable",
    "get_mcp_client",
    "wrap_mcp_tools_for_logging",
]
