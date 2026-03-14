import os
import glob
from pathlib import Path
from typing import List, Dict, Any

class LocalTools:
    """
    Tools that the Coder model can use to edit files locally.
    """
    def __init__(self, workspace_path: str = "."):
        self.workspace_path = Path(workspace_path).resolve()

    def read_file(self, relative_path: str) -> str:
        """Read the contents of a local file."""
        target_path = (self.workspace_path / relative_path).resolve()
        
        # Security: Prevent escaping the workspace
        if not str(target_path).startswith(str(self.workspace_path)):
            return "Error: Cannot read outside of workspace."
            
        if not target_path.exists():
            return f"Error: File '{relative_path}' not found."
            
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def write_file(self, relative_path: str, content: str) -> str:
        """Write content to a local file."""
        target_path = (self.workspace_path / relative_path).resolve()
        
        # Security: Prevent escaping the workspace
        if not str(target_path).startswith(str(self.workspace_path)):
            return "Error: Cannot write outside of workspace."
            
        try:
            # Ensure directory exists
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Success: Wrote to '{relative_path}'."
        except Exception as e:
            return f"Error writing file: {str(e)}"

    def list_directory(self, relative_path: str = ".") -> str:
        """List contents of a directory."""
        target_path = (self.workspace_path / relative_path).resolve()
        
        # Security: Prevent escaping the workspace
        if not str(target_path).startswith(str(self.workspace_path)):
            return "Error: Cannot list outside of workspace."
            
        if not target_path.exists() or not target_path.is_dir():
            return f"Error: Directory '{relative_path}' not found."
            
        try:
            entries = os.listdir(target_path)
            return "\n".join(entries) if entries else "Directory is empty."
        except Exception as e:
            return f"Error listing directory: {str(e)}"
