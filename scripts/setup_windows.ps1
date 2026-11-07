# Creates .venv with CUDA PyTorch on Windows. Run from the repository root.
param([string]$PythonVersion = "3.11", [string]$CudaIndex = "cu128")
$ErrorActionPreference = "Stop"

py -$PythonVersion -m venv .venv
if ($LASTEXITCODE -ne 0) { throw "Python $PythonVersion was not found. Install it from python.org first." }

$python = ".\.venv\Scripts\python.exe"
& $python -m pip install --upgrade pip
& $python -m pip install torch --index-url "https://download.pytorch.org/whl/$CudaIndex"
if ($LASTEXITCODE -ne 0) { throw "PyTorch install failed." }
& $python -m pip install -e ".[dev]"
if ($LASTEXITCODE -ne 0) { throw "Project install failed." }

& $python -c "import torch; assert torch.cuda.is_available(), \"CUDA is not available\"; print(torch.__version__, torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), \"GiB\")"
if ($LASTEXITCODE -ne 0) { throw "CUDA check failed. Update the NVIDIA driver and run nvidia-smi." }

& $python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "Unit tests failed." }
Write-Host "Setup complete."
