# Start the Autonomous Test Orchestration Agent locally (API + Streamlit UI).
#
# Usage:   .\run_local.ps1          # start both services
#          .\run_local.ps1 -ApiOnly # start only the REST API
#
# The project needs Python 3.11/3.12 - Playwright wheels lag on 3.13+ - so the
# interpreter is pinned to the .venv311 environment created for that reason.
param([switch]$ApiOnly)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot '.venv311\Scripts\python.exe'

if (-not (Test-Path $py)) {
    Write-Error "Missing .venv311. Create it with:`n  uv venv --python 3.11 .venv311`n  `$env:VIRTUAL_ENV='.venv311'; uv pip install -r requirements.txt`n  .venv311\Scripts\python.exe -m playwright install chromium"
}

# A stale GROQ_API_KEY inherited from the shell silently beats .env, because
# config.py calls load_dotenv(override=False). Clear it so .env is authoritative.
if ($env:GROQ_API_KEY) {
    Write-Host "Clearing inherited GROQ_API_KEY so .env wins." -ForegroundColor Yellow
    Remove-Item Env:\GROQ_API_KEY
}

Start-Process -FilePath $py -ArgumentList '-m','uvicorn','api.app:app','--host','127.0.0.1','--port','8000'
Write-Host "API -> http://127.0.0.1:8000/health"

if (-not $ApiOnly) {
    Start-Process -FilePath $py -ArgumentList '-m','streamlit','run','ui/streamlit_app.py','--server.port','8501'
    Write-Host "UI  -> http://localhost:8501"
}
