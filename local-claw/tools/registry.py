"""
tools/registry.py — Tool registry and dispatcher for NeuroClaw Bot.
Inspired by Claude Code's tool system and OpenClaw's first-class tools.

Each tool has a name, a function, and an argument schema.
The agent loop calls tools by name and passes parsed JSON arguments.
"""
import os
import subprocess
from pathlib import Path
from typing import Callable, Any


# Commands that are always blocked for safety
BLOCKED_COMMANDS = [
    "rm -rf /", "rm -rf ~", "del /f /s /q c:\\",
    "format c:", "shutdown", "mkfs", "dd if=",
    ":(){ :|:& };:"  # fork bomb
]

# File extensions recognized as source code
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go",
    ".rs", ".cpp", ".c", ".h", ".cs", ".rb", ".php",
    ".swift", ".kt", ".vue", ".html", ".css", ".scss",
    ".json", ".yaml", ".yml", ".toml", ".sh", ".bash",
    ".md", ".txt", ".env.example"
}

# Directories to skip when walking the project
SKIP_DIRS = {
    "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist", "build", ".git", ".agent_index", ".idea", ".vscode"
}


class ToolRegistry:
    """
    Maintains a registry of callable tools.
    The agent loop calls `call(name, args)` which dispatches to the right function.
    """

    def __init__(self, project_root: str):
        self.project_root = Path(project_root).resolve()
        self._tools: dict[str, dict[str, Any]] = {}
        self._register_builtin_tools()

    # ──────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────

    def register(self, name: str, func: Callable, description: str = ""):
        """Register a new tool."""
        self._tools[name] = {"fn": func, "description": description}

    def call(self, name: str, args: dict) -> str:
        """Call a tool by name with the given arguments."""
        if name not in self._tools:
            available = ", ".join(self._tools.keys())
            return f"[ERROR] Unknown tool '{name}'. Available tools: {available}"
        try:
            return str(self._tools[name]["fn"](**args))
        except TypeError as e:
            return f"[ERROR] Tool '{name}' called with wrong arguments: {e}"
        except Exception as e:
            return f"[ERROR] Tool '{name}' raised an exception: {type(e).__name__}: {e}"

    @property
    def schema_description(self) -> str:
        """Returns a human-readable description of all available tools."""
        lines = ["AVAILABLE TOOLS:"]
        for name, info in self._tools.items():
            lines.append(f"- {name}: {info['description']}")
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────
    # Built-in Tool Definitions
    # ──────────────────────────────────────────────────────

    def _register_builtin_tools(self):
        root = self.project_root

        # ── read_file ──────────────────────────────────────
        def read_file(path: str) -> str:
            """Read a file and return its contents."""
            target = (root / path).resolve()
            # Security: prevent path traversal outside project
            try:
                target.relative_to(root)
            except ValueError:
                return f"[BLOCKED] Path '{path}' is outside the project directory."
            if not target.exists():
                return f"[ERROR] File not found: {path}"
            if not target.is_file():
                return f"[ERROR] '{path}' is a directory, not a file."
            try:
                content = target.read_text(encoding="utf-8", errors="ignore")
                if len(content) > 8000:
                    return content[:8000] + f"\n\n...[TRUNCATED — file has {len(content)} chars total]"
                return content
            except Exception as e:
                return f"[ERROR] Could not read file: {e}"

        # ── write_file ─────────────────────────────────────
        def write_file(path: str, content: str) -> str:
            """Write content to a file, creating parent directories if needed."""
            target = (root / path).resolve()
            try:
                target.relative_to(root)
            except ValueError:
                return f"[BLOCKED] Path '{path}' is outside the project directory."
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"[OK] Written {len(content)} chars to {path}"

        # ── list_files ─────────────────────────────────────
        def list_files(directory: str = ".") -> str:
            """List all files in a directory (recursively, skipping ignored dirs)."""
            target = (root / directory).resolve()
            if not target.exists():
                return f"[ERROR] Directory not found: {directory}"
            results = []
            for entry in sorted(target.rglob("*")):
                # Skip ignored directories
                if any(part in SKIP_DIRS for part in entry.parts):
                    continue
                if entry.name.startswith("."):
                    continue
                if entry.is_file():
                    rel = entry.relative_to(root)
                    size = entry.stat().st_size
                    results.append(f"{rel}  ({size} bytes)")
                    if len(results) >= 300:
                        results.append("... (truncated at 300 files)")
                        break
            return "\n".join(results) if results else "[Empty directory]"

        # ── search_code ────────────────────────────────────
        def search_code(query: str, directory: str = ".", extensions: str = "") -> str:
            """Search for a string pattern in code files. Returns matching lines with file:line context."""
            target = (root / directory).resolve()
            ext_filter = set(extensions.split(",")) if extensions else CODE_EXTENSIONS
            matches = []
            for fpath in target.rglob("*"):
                if any(part in SKIP_DIRS for part in fpath.parts):
                    continue
                if not fpath.is_file():
                    continue
                if fpath.suffix.lower() not in ext_filter:
                    continue
                try:
                    for i, line in enumerate(fpath.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                        if query.lower() in line.lower():
                            rel = fpath.relative_to(root)
                            matches.append(f"{rel}:{i}:  {line.strip()}")
                            if len(matches) >= 40:
                                matches.append("... (truncated at 40 results)")
                                return "\n".join(matches)
                except Exception:
                    pass
            return "\n".join(matches) if matches else f"[No matches found for '{query}']"

        # ── run_terminal_command ───────────────────────────
        def run_terminal_command(command: str) -> str:
            """Execute a shell command in the project directory. Output is capped at 3000 chars."""
            cmd_lower = command.lower()
            for blocked in BLOCKED_COMMANDS:
                if blocked.lower() in cmd_lower:
                    return f"[BLOCKED] Dangerous command refused: '{command}'"

            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    cwd=str(root),
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                stdout = result.stdout or ""
                stderr = result.stderr or ""
                # Cap outputs
                if len(stdout) > 2500:
                    stdout = stdout[:2500] + "\n...[stdout truncated]"
                if len(stderr) > 500:
                    stderr = stderr[:500] + "\n...[stderr truncated]"
                return (
                    f"[stdout]\n{stdout}"
                    f"\n[stderr]\n{stderr}"
                    f"\n[exit code: {result.returncode}]"
                )
            except subprocess.TimeoutExpired:
                return "[ERROR] Command timed out after 60 seconds."
            except Exception as e:
                return f"[ERROR] Failed to run command: {e}"

        # ── Register all tools ─────────────────────────────
        self.register("read_file", read_file,
                      "read_file(path: str) → Read a file's full content")
        self.register("write_file", write_file,
                      "write_file(path: str, content: str) → Create or overwrite a file")
        self.register("list_files", list_files,
                      "list_files(directory: str = '.') → List all project files recursively")
        self.register("search_code", search_code,
                      "search_code(query: str, directory: str = '.') → Search for a string in code files")
        self.register("run_terminal_command", run_terminal_command,
                      "run_terminal_command(command: str) → Run a shell command (e.g., pytest, pip, git)")
