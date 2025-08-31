from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from langchain_core.callbacks.base import BaseCallbackHandler  # type: ignore


class ToolLogHandler(BaseCallbackHandler):
    """Log tool start/end with truncated inputs/outputs.

    Safe for LangGraph/LCEL by using callbacks instead of wrapping tools.
    Enable by passing in config={"callbacks":[ToolLogHandler(...)]}.
    """

    def __init__(self, logger: Optional[logging.Logger] = None, max_len: int = 800) -> None:
        self.log = logger or logging.getLogger("mcp.tools")
        self.max_len = max_len

    def _trunc(self, obj: Any) -> str:
        try:
            s = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
        except Exception:
            s = str(obj)
        if len(s) > self.max_len:
            return s[: self.max_len - 3] + "..."
        return s

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        name = serialized.get("name") or serialized.get("id") or "(tool)"
        try:
            self.log.info("MCP tool start: %s input=%s", name, self._trunc(input_str))
        except Exception:
            pass

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        try:
            self.log.info("MCP tool end: output=%s", self._trunc(output))
        except Exception:
            pass

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:  # type: ignore[override]
        try:
            self.log.error("MCP tool error: %s", error)
        except Exception:
            pass
