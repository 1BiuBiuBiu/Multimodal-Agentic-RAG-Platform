$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
Set-Location -LiteralPath $PSScriptRoot

$pythonPath = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Missing .venv. Run: python -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
}
if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot ".env"))) {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot ".env.example") -Destination (Join-Path $PSScriptRoot ".env")
    throw "Created .env. Set your DEEPSEEK_API_KEY there and run this script again."
}

$runtimePath = Join-Path $PSScriptRoot ".runtime"
New-Item -ItemType Directory -Path $runtimePath -Force | Out-Null

function Start-ServiceProcess {
    param([string]$Name, [int]$Port, [string[]]$Arguments, [string]$HealthUrl)
    $pidFile = Join-Path $runtimePath "$Name.pid"
    $listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($listener) {
        if (Test-Path -LiteralPath $pidFile) {
            $savedPid = [int](Get-Content -LiteralPath $pidFile)
            $ownedPids = @($savedPid)
            $ownedPids += @(Get-CimInstance Win32_Process -Filter "ParentProcessId = $savedPid" | Select-Object -ExpandProperty ProcessId)
            if (@($listener.OwningProcess | Where-Object { $ownedPids -contains $_ }).Count -gt 0) {
                Write-Host "$Name is already running: $HealthUrl"
                return
            }
        }
        throw "Port $Port is already occupied. No existing process was stopped."
    }
    $stdoutPath = Join-Path $runtimePath "$Name.out.log"
    $stderrPath = Join-Path $runtimePath "$Name.err.log"
    $serviceProcess = Start-Process -FilePath $pythonPath -ArgumentList $Arguments `
        -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
    Set-Content -LiteralPath $pidFile -Value $serviceProcess.Id
    for ($attempt = 0; $attempt -lt 90; $attempt++) {
        if ($serviceProcess.HasExited) {
            Get-Content -LiteralPath $stderrPath -Tail 30
            throw "$Name exited. See $stderrPath"
        }
        try {
            $response = Invoke-WebRequest -Uri $HealthUrl -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -eq 200) {
                Write-Host "$Name started (PID $($serviceProcess.Id))."
                return
            }
        } catch { }
        Start-Sleep -Seconds 1
    }
    throw "$Name is still starting. Check $stderrPath"
}

Start-ServiceProcess -Name "api" -Port 8000 `
    -Arguments @("-m", "uvicorn", "app.api:app", "--host", "127.0.0.1", "--port", "8000") `
    -HealthUrl "http://127.0.0.1:8000/health"
Start-ServiceProcess -Name "ui" -Port 8501 `
    -Arguments @("-m", "streamlit", "run", "app/main.py", "--server.address", "127.0.0.1", "--server.port", "8501", "--server.headless", "true", "--browser.gatherUsageStats", "false") `
    -HealthUrl "http://127.0.0.1:8501/_stcore/health"
Write-Host "UI: http://127.0.0.1:8501"
Write-Host "API docs: http://127.0.0.1:8000/docs"
Write-Host "Stop: powershell -ExecutionPolicy Bypass -File .\stop.ps1"
