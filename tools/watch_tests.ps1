<#
PowerShell helper to run a live test watcher for this project.
Usage: Open PowerShell in the repo root and run `./tools/watch_tests.ps1`.
This script will install dev requirements (if missing) and start `ptw` (pytest-watch).
#>

$ErrorActionPreference = 'Stop'

function Ensure-DevDeps {
    if (-not (Get-Command ptw -ErrorAction SilentlyContinue)) {
        Write-Host "pytest-watch not found, installing dev requirements..."
        python -m pip install -r requirements-dev.txt
    }
}

Ensure-DevDeps

Write-Host "Starting pytest-watch (ptw). Press Ctrl+C to stop."
ptw -q
