# rerun_all_notebooks.ps1
# 1. Re-executes all T*.ipynb on the academic branch (Windows Python = Microsoft YaHei, CJK+Latin perfect)
# 2. Commits & pushes academic branch
# 3. Merges academic -> main and pushes main
# Run from D:\Git\EconometricsML in PowerShell.

$PYTHON = "d:\anaconda3\envs\interview_prep\python.exe"
$NB_DIR  = "$PSScriptRoot\notebooks\traditional"
Set-Location $PSScriptRoot

# ── Step 1: stash any local changes, switch to academic ───────────────────────
Write-Host ""
Write-Host "=== Stashing local changes and switching to academic ===" -ForegroundColor Cyan
$stashed = $false
$dirty = git status --porcelain
if ($dirty) {
    git stash push -m "pre-rerun stash"
    $stashed = $true
}
git checkout academic
if ($stashed) { git stash pop }

# ── Step 2: re-execute all T*.ipynb with Windows fonts ───────────────────────
Write-Host ""
Write-Host "=== Re-executing traditional notebooks with Windows fonts ===" -ForegroundColor Cyan

$notebooks = Get-ChildItem "$NB_DIR\T*.ipynb" | Sort-Object Name
foreach ($nb in $notebooks) {
    Write-Host "  Executing: $($nb.Name) ..." -ForegroundColor Yellow
    & $PYTHON -m nbcon