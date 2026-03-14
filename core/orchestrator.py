import json
import os
import re
from pathlib import Path
from model.model_router import ModelRouter, ModelRole
from core.tools import LocalTools

class MultiAgentOrchestrator:
    """
    Orchestrates complex multi-step tasks.
    Planner breaks down tasks. Coder executes them.
    Adheres strictly to <50% resource constraints via ModelRouter's hot-swapping.
    """
    def __init__(self, router: ModelRouter, workspace_path: str = "."):
        self.router = router
        self.tools = LocalTools(workspace_path)
        self.memory_path = Path(workspace_path) / "memory.json"
        
    def _save_memory(self, steps: list, current_step_index: int):
        with open(self.memory_path, "w") as f:
            json.dump({"steps": steps, "current_step_index": current_step_index}, f)
            
    def _load_memory(self) -> dict:
        if self.memory_path.exists():
            with open(self.memory_path, "r") as f:
                return json.load(f)
        return {"steps": [], "current_step_index": 0}
        
    def _clear_memory(self):
        if self.memory_path.exists():
            os.remove(self.memory_path)
            
    def run_advanced_task(self, prompt: str):
        """Main entry point for advanced multi-step execution."""
        print("\n" + "="*50)
        print("🚀 Starting Advanced Multi-Agent Orchestration")
        print("="*50)
        
        # 1. Planning Phase
        print("\n🧠 [Phase 1: Planning] Waking up Planner Model...")
        planner = self.router._pick_loader(ModelRole.PLANNER)
        
        plan_prompt = f"""You are the Master Planner. 
The user has requested: "{prompt}"

Break this down into explicit, numbered steps. Be concise. Reply ONLY with the numbered steps.
Example:
1. Read file main.py
2. Write file debug.py with print statements
"""
        plan_text = planner.generate(plan_prompt, max_tokens=500)
        
        # Parse steps
        steps = []
        for line in plan_text.split('\n'):
            line = line.strip()
            # Match "1. " or "1) " or just "Step 1:"
            if re.match(r'^(\d+[\.\)]|Step \d+:?)\s+', line, re.IGNORECASE):
                # Clean up the prefix
                cleaned = re.sub(r'^(\d+[\.\)]|Step \d+:?)\s+', '', line, flags=re.IGNORECASE)
                if cleaned:
                    steps.append(cleaned)
                    
        # Fallback if regex failed
        if not steps:
             steps = [x.strip() for x in plan_text.split('\n') if x.strip()]
             
        if not steps:
            print("❌ Planner failed to generate steps. Aborting.")
            return

        print("\n📝 Generated Plan:")
        for i, step in enumerate(steps):
             print(f"  {i+1}. {step}")
             
        self._save_memory(steps, 0)
        
        # 2. Execution Phase
        print("\n⚙️ [Phase 2: Execution] Waking up Coder Model...")
        coder = self.router._pick_loader(ModelRole.CODER)
        
        memory = self._load_memory()
        
        for i in range(memory["current_step_index"], len(memory["steps"])):
            step = memory["steps"][i]
            print(f"\n▶️ Executing Step {i+1}: {step}")
            
            exec_prompt = f"""You are an Expert Coder executing a step in a larger plan.
Current Step: "{step}"
Available Tools:
- read_file(path) : Returns file contents.
- write_file(path, content) : Writes content to file.
- list_directory(path) : Lists files in directory.

If you need to use a tool, reply ONLY with the tool command.
Example: write_file("test.py", "print('hello')")
If you have finished the step, reply with: [DONE]
"""
            # Allow multiple tool calls per step
            max_attempts = 5
            attempts = 0
            while attempts < max_attempts:
                response = coder.generate(exec_prompt, max_tokens=500).strip()
                print(f"  Coder: {response}")
                
                if "[DONE]" in response.upper():
                    print(f"✅ Step {i+1} completed.")
                    break
                    
                # Basic tool parsing
                if "write_file(" in response:
                    try:
                        # Very simplified parsing
                        parts = response.split('write_file(', 1)[1].rsplit(')', 1)[0]
                        # Assume roughly: "filename.py", "content"
                        # This is a naive split and will break on complex quotes, but works for demo
                        args = [p.strip().strip('\'"') for p in parts.split(",", 1)]
                        if len(args) == 2:
                            res = self.tools.write_file(args[0], args[1])
                            print(f"  🛠️ Tool Result: {res}")
                            exec_prompt += f"\nTool Result: {res}\nWhat next? (Reply [DONE] if finished)"
                        else:
                            exec_prompt += "\nTool Result: Error parsing write_file arguments."
                    except Exception as e:
                         exec_prompt += f"\nTool Result: Error parsing command: {e}"
                elif "read_file(" in response:
                    try:
                         path = response.split('read_file(', 1)[1].rsplit(')', 1)[0].strip().strip('\'"')
                         res = self.tools.read_file(path)
                         # Truncate to avoid blowing context
                         if len(res) > 2000: res = res[:2000] + "...[TRUNCATED]"
                         print(f"  🛠️ Tool Result: Read {len(res)} chars")
                         exec_prompt += f"\nFile Contents:\n{res}\nWhat next? (Reply [DONE] if finished)"
                    except Exception as e:
                         exec_prompt += f"\nTool Result: Error parsing command: {e}"
                elif "list_directory(" in response:
                    try:
                         path = response.split('list_directory(', 1)[1].rsplit(')', 1)[0].strip().strip('\'"')
                         if not path: path = "."
                         res = self.tools.list_directory(path)
                         print(f"  🛠️ Tool Result: Listed directory")
                         exec_prompt += f"\nDirectory Contents:\n{res}\nWhat next? (Reply [DONE] if finished)"
                    except Exception as e:
                         exec_prompt += f"\nTool Result: Error parsing command: {e}"
                else:
                    # Coder is just talking, ask it if it's done
                    exec_prompt += f"\nAgent Output: {response}\nUse a tool or reply [DONE]"
                attempts += 1
                
            # Save progress
            self._save_memory(memory["steps"], i + 1)
            
        print("\n🎉 All steps completed successfully!")
        self._clear_memory()
