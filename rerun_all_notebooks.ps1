# rerun_all_notebooks.ps1
# 1. Re-executes all T*.ipynb on the academic branch (Windows Python = Microsoft YaHei, CJK+Latin perfect)
# 2. Commits & pushes academic branch (notebooks/traditional + notebooks/academic)
# 3. Merges academic -> main and pushes main
# Run from D:\Git\EconometricsML in PowerShell.

$PYTHON = "d:\anaconda3\envs\interview_prep\python.exe"
$NB_DIR  = "$PSScriptRoot\notebooks\traditional"

# ── Step 1: make sure we are on academic ──────────────────────────────────────
Write-Host ""
Write-Host "=== Switching to academic branch ===" -ForegroundColor Cyan
Set-Location $PSScriptRoot
git checkout academic

# ── Step 2: re-execute all T*.ipynb with Windows fonts ───────────────────────
Write-Host ""
Write-Host "=== Re-executing traditional notebooks with Windows fonts ===" -ForegroundColor Cyan

$notebooks = Get-ChildItem "$NB_DIR\T*.ipynb" | Sort-Object Name
foreach ($nb in $notebooks) {
    Write-Host "  Executing: $($nb.Name) ..." -ForegroundColor Yellow
    & $PYTHON -m nbconvert --to notebook --execute --inplace `
        --ExecutePreprocessor.timeout=300 `
        $nb.FullName
    if ($LASTEXITCODE -eq 0