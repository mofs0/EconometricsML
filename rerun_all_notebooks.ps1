# rerun_all_notebooks.ps1
# Re-executes all T*.ipynb notebooks using the local Windows Python interpreter
# so that Microsoft YaHei font renders both CJK and Latin characters perfectly.
# Run this from D:\Git\EconometricsML in PowerShell.

$PYTHON = "d:\anaconda3\envs\interview_prep\python.exe"
$NB_DIR = "$PSScriptRoot\notebooks\traditional"

Write-Host "=== Re-executing all notebooks with Windows fonts ===" -ForegroundColor Cyan

$notebooks = Get-ChildItem "$NB_DIR\T*.ipynb" | Sort-Object Name

foreach ($nb in $notebooks) {
    Write-Host "Executing: $($nb.Name) ..." -ForegroundColor Yellow
    & $PYTHON -m nbconvert --to notebook --execute --inplace `
        --ExecutePreprocessor.timeout=300 `
        $nb.FullName
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  OK" -ForegroundColor Green
    } else {
        Write-Host "  FAILED (exit $LASTEXITCODE)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "=== Committing and pushing to GitHub ===" -ForegroundColor Cyan
Set-Location $PSScriptRoot
git add "notebooks\traditional\T*.ipynb"
git commit -m "re-execute all notebooks with Windows fonts (CJK + Latin render perfectly)"
git push origin main

Write-Host ""
Write-Host "Done! Check GitHub to verify all plots look correct." -ForegroundColor Green
