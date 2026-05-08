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
    & $PYTHON -m nbconvert --to notebook --execute --inplace `
        --ExecutePreprocessor.timeout=300 `
        $nb.FullName
    if ($LASTEXITCODE -eq 0) {
        Write-Host "    OK" -ForegroundColor Green
    } else {
        Write-Host "    FAILED (exit $LASTEXITCODE)" -ForegroundColor Red
    }
}

# ── Step 3: commit & push academic branch ────────────────────────────────────
Write-Host ""
Write-Host "=== Committing and pushing academic branch ===" -ForegroundColor Cyan
git add -A
git commit -m "re-execute T* notebooks with Windows fonts (Microsoft YaHei)" --allow-empty
git push origin academic

# ── Step 4: merge academic -> main and push main ──────────────────────────────
Write-Host ""
Write-Host "=== Merging academic into main and pushing ===" -ForegroundColor Cyan
git checkout main
git pull origin main --rebase          # sync with remote first to avoid rejection
git merge academic --no-ff -m "merge academic: A01-A07 paper writing guide + CJK font fixes"
git push origin main

Write-Host ""
Write-Host "=== All done! ===" -ForegroundColor Green
Write-Host "Both 'main' and 'academic' are now up to date on GitHub." -ForegroundColor Green
