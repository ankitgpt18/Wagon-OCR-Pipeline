"""
Server startup script that ensures NVIDIA CUDA DLLs are loaded
and PaddlePaddle is fully initialized BEFORE uvicorn imports the app.
This prevents the circular import / partial module initialization errors
on Windows with paddlepaddle-gpu.
"""
import os
import sys
import site
from pathlib import Path

# Step 1: Add NVIDIA CUDA DLL directories to the search path (Windows only)
if os.name == "nt":
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        nvidia_path = Path(sp) / "nvidia"
        if nvidia_path.exists():
            for p in nvidia_path.rglob("bin"):
                try:
                    os.add_dll_directory(str(p))
                except Exception:
                    pass

# Step 2: Pre-import torch and paddle to prevent DLL collision and ensure fully initialized sys.modules
try:
    import torch
    print(f"PyTorch loaded: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"PyTorch import failed: {e}")

try:
    import paddle
    print(f"PaddlePaddle loaded: {paddle.__version__} | CUDA compiled: {paddle.is_compiled_with_cuda()}")
except Exception as e:
    print(f"PaddlePaddle import failed: {e} - OCR will run on CPU")

# Step 3: Start uvicorn
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "cris_ocr_pipeline.scripts.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
