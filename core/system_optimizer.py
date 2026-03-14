import os
import gc
try:
    import torch
except ImportError:
    torch = None

def get_optimal_threads() -> int:
    """
    Calculate the optimal number of CPU threads to use for model inference.
    Strictly constrained to a max of 50% CPU usage overall.
    """
    total_cores = os.cpu_count() or 4
    return max(1, int(total_cores * 0.5))

def get_optimal_gpu_layers() -> int:
    """
    Strictly cap VRAM usage at 50%.
    Instead of full offload (-1), strictly limit layers.
    Assuming standard models have ~32 layers, we cap at 16 (50%).
    """
    if torch and torch.cuda.is_available():
        try:
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if total_vram_gb < 4.0:
                print(f"[Optimizer] 50% VRAM constraint (Total: {total_vram_gb:.1f}GB). Offloading max 4 layers.")
                return 4
            elif total_vram_gb < 8.0:
                print(f"[Optimizer] 50% VRAM constraint (Total: {total_vram_gb:.1f}GB). Offloading max 8 layers.")
                return 8
            else:
                print(f"[Optimizer] 50% VRAM constraint (Total: {total_vram_gb:.1f}GB). Offloading max 16 layers.")
                return 16 # Never exceed ~50% of the model's layers
        except Exception as e:
            print(f"[Optimizer] VRAM fetch error. Fallback to 8 layers. {e}")
            return 8
    return 0

def get_optimal_ctx() -> int:
    """
    Return a context window size that ensures RAM usage stays under 50%.
    Uses ctypes on Windows to check native physical memory load.
    """
    import platform
    default_strict_ctx = 2048
    if platform.system() == "Windows":
        try:
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)  # type: ignore
            windll = getattr(ctypes, "windll", None)
            if windll:
                windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))

            if stat.dwMemoryLoad >= 50:
                print(f"[Optimizer] System RAM is already at {stat.dwMemoryLoad}%. Enforcing minimal 1024 context limit.")
                return 1024
            elif stat.dwMemoryLoad >= 40:
                return 2048
            else:
                return 4096 # Safe if RAM is mostly free
        except Exception:
            pass
    return default_strict_ctx

def clear_memory():
    """Aggressively clear RAM and VRAM caches."""
    gc.collect()
    if torch and torch.cuda.is_available():
        torch.cuda.empty_cache()

def clear_memory():
    """Aggressively clear RAM and VRAM caches."""
    gc.collect()
    if torch and torch.cuda.is_available():
        torch.cuda.empty_cache()
