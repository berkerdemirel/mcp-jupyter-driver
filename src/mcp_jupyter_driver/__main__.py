"""Entrypoint: `python -m mcp_jupyter_driver` or `mcp-jupyter-driver`.

Run with `--self-check` to import all modules and confirm the FastMCP app
constructs without errors, without actually starting a kernel.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(prog="mcp-jupyter-driver")
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Import everything and print OK, then exit.",
    )
    args = parser.parse_args()

    if args.self_check:
        from . import execution, registry, server, session  # noqa: F401
        from .server import mcp

        print(f"mcp-jupyter-driver: {mcp.name} ready")
        print(f"tools: {sorted(t.name for t in mcp._tool_manager.list_tools())}")
        return 0

    from .server import mcp

    mcp.run(transport="stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
