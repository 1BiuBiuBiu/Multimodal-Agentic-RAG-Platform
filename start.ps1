param(
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
Set-Location -LiteralPath $PSScriptRoot

$pythonPath = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Missing .venv. Run: python -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
}

# Verify venv python can load core DLLs; fallback to base Python if Windows stub DLL issue occurs
$venvWorks = $false
try {
    $proc = Start-Process -FilePath $pythonPath -ArgumentList @("-c", "import ssl") -PassThru -WindowStyle Hidden -Wait
    if ($proc.ExitCode -eq 0) {
        $venvWorks = $true
    }
} catch { }

if (-not $venvWorks) {
    $globalPy = $null
    $pyCmd = Get-Command python -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($pyCmd -and $pyCmd.Source -and ($pyCmd.Source -ne $pythonPath) -and ($pyCmd.Source -notlike "*WindowsApps*")) {
        $globalPy = $pyCmd.Source
    }
    if (-not $globalPy -or -not (Test-Path -LiteralPath $globalPy)) {
        $fallbackPy = "C:\Users\33983\AppData\Local\Programs\Python\Python312\python.exe"
        if (Test-Path -LiteralPath $fallbackPy) {
            $globalPy = $fallbackPy
        }
    }
    if ($globalPy) {
        $pythonPath = $globalPy
        $sitePackages = Join-Path $PSScriptRoot ".venv\Lib\site-packages"
        if ($env:PYTHONPATH) {
            $env:PYTHONPATH = "$sitePackages;$PSScriptRoot;$env:PYTHONPATH"
        } else {
            $env:PYTHONPATH = "$sitePackages;$PSScriptRoot"
        }
        Write-Host "Using base Python with virtual environment libraries: $pythonPath"
    }
}

if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot ".env"))) {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot ".env.example") -Destination (Join-Path $PSScriptRoot ".env")
    throw "Created .env. Set your DEEPSEEK_API_KEY there and run this script again."
}

$runtimePath = Join-Path $PSScriptRoot ".runtime"
New-Item -ItemType Directory -Path $runtimePath -Force | Out-Null

function Test-HealthUrl {
    param([string]$HealthUrl)
    try {
        $response = Invoke-WebRequest -Uri $HealthUrl -UseBasicParsing -TimeoutSec 2
        return ($response.StatusCode -eq 200)
    } catch {
        return $false
    }
}

function Start-ServiceProcess {
    param([string]$Name, [int]$Port, [string[]]$Arguments, [string]$HealthUrl, [string]$ExpectedModule)
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
        foreach ($ownerPid in ($listener.OwningProcess | Select-Object -Unique)) {
            $ownerProc = Get-CimInstance Win32_Process -Filter "ProcessId = $ownerPid" -ErrorAction SilentlyContinue
            if ($ownerProc -and $ownerProc.Name -like "python*" -and $ownerProc.CommandLine -like "*$ExpectedModule*" -and (Test-HealthUrl -HealthUrl $HealthUrl)) {
                Set-Content -LiteralPath $pidFile -Value $ownerPid
                Write-Host "$Name is already running (PID $ownerPid): $HealthUrl"
                return
            }
        }
        throw "Port $Port is already occupied by another process. No existing process was stopped."
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
        if (Test-HealthUrl -HealthUrl $HealthUrl) {
            Write-Host "$Name started (PID $($serviceProcess.Id))."
            return
        }
        Start-Sleep -Seconds 1
    }
    throw "$Name is still starting. Check $stderrPath"
}

Start-ServiceProcess -Name "api" -Port 8000 `
    -Arguments @("-m", "uvicorn", "app.api:app", "--host", "127.0.0.1", "--port", "8000") `
    -HealthUrl "http://127.0.0.1:8000/health" `
    -ExpectedModule "uvicorn app.api:app"

Start-ServiceProcess -Name "ui" -Port 8501 `
    -Arguments @("-m", "streamlit", "run", "app/main.py", "--server.address", "127.0.0.1", "--server.port", "8501", "--server.headless", "true", "--browser.gatherUsageStats", "false") `
    -HealthUrl "http://127.0.0.1:8501/_stcore/health" `
    -ExpectedModule "streamlit run app/main.py"

$uiUrl = "http://127.0.0.1:8501"
Write-Host "UI: $uiUrl"
Write-Host "API docs: http://127.0.0.1:8000/docs"
Write-Host "Stop: powershell -ExecutionPolicy Bypass -File .\stop.ps1"

if (-not $NoBrowser) {
    Write-Host "Opening UI in default browser: $uiUrl"
    Start-Process $uiUrl
}
