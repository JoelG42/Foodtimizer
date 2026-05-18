# Launch Foodtimizer tracker in a NEW PowerShell window.
# Your current console stays free for git, pytest, foodtimizer CLI, etc.
#
# Usage (from anywhere):
#   powershell -ExecutionPolicy Bypass -File C:\Foodtimizer\scripts\start-tracker.ps1
#
# Or double-click start-tracker.bat in Explorer.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Track = Join-Path $Root ".venv\Scripts\foodtimizer-track.exe"
$Config = Join-Path $Root "examples\day.yaml"
$Logs = Join-Path $Root "logs"

if (-not (Test-Path $Track)) {
    Write-Host "foodtimizer-track not found. Run once:" -ForegroundColor Yellow
    Write-Host "  cd $Root"
    Write-Host '  .\.venv\Scripts\python.exe -m pip install -e ".[app]"'
    exit 1
}

$cmd = @"
Set-Location '$Root'
Write-Host 'Foodtimizer tracker — close this window or press Ctrl+C to stop.' -ForegroundColor Cyan
Write-Host 'Logs: $Logs' -ForegroundColor DarkGray
& '$Track' --config '$Config' --logs-dir '$Logs'
Write-Host ''
Write-Host 'Tracker stopped. You can close this window.' -ForegroundColor Green
pause
"@

Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-Command", $cmd
)

Write-Host "Tracker starting in a new window..." -ForegroundColor Green
