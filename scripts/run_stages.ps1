# Runs stages 1-3 and writes results under results\<ModelName>. Run from the repository root.
param(
    [string]$Model = "Qwen/Qwen2.5-1.5B-Instruct",
    [ValidateSet("all", "1", "2", "3", "3int8", "gsm8k")][string]$Stage = "all",
    [int]$GsmSamples = 200
)
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$python = ".\.venv\Scripts\python.exe"
$name = $Model.Split("/")[-1]

function Invoke-Stage([string]$Script, [string]$Dir, [string[]]$Extra = @()) {
    Write-Host "=== $Script ==="
    & $python "experiments\$Script" --model $Model --device cuda --out "results\$name\$Dir" @Extra
    if ($LASTEXITCODE -ne 0) { throw "$Script failed." }
}

if ($Stage -in "all", "1") { Invoke-Stage "stage1_kv_distribution.py" "stage1" }
if ($Stage -in "all", "2") { Invoke-Stage "stage2_operator.py" "stage2" }
if ($Stage -in "all", "3") { Invoke-Stage "stage3_model.py" "stage3" }
if ($Stage -in "all", "3int8") { Invoke-Stage "stage3_model.py" "stage3_int8prefill" @("--prefill", "int8", "--configs", "per_token_head:dequant,per_tensor:dequant") }
if ($Stage -in "all", "gsm8k") { Invoke-Stage "stage3_gsm8k.py" "stage3" @("--samples", "$GsmSamples") }
Write-Host "Done. Results are in results\$name"
