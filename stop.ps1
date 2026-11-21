$ErrorActionPreference = "Stop"
$runtimePath = Join-Path $PSScriptRoot ".runtime"

$services = @(
    @{ Name = "ui";  Port = 8501; ExpectedModule = "streamlit run app/main.py" },
    @{ Name = "api"; Port = 8000; ExpectedModule = "uvicorn app.api:app" }
)

foreach ($svc in $services) {
    $name = $svc.Name
    $port = $svc.Port
    $expectedModule = $svc.ExpectedModule
    $pidFile = Join-Path $runtimePath "$name.pid"
    $stopped = $false

    if (Test-Path -LiteralPath $pidFile) {
        $rawPid = (Get-Content -LiteralPath $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
        if ($rawPid -and ($rawPid -match '^\d+$')) {
            $savedPid = [int]$rawPid
            $serviceProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $savedPid" -ErrorAction SilentlyContinue
            if ($serviceProcess) {
                if ($serviceProcess.Name -like "python*" -and $serviceProcess.CommandLine -like "*$expectedModule*") {
                    $workers = Get-CimInstance Win32_Process -Filter "ParentProcessId = $savedPid"
                    foreach ($worker in $workers) {
                        if ($worker.Name -like "python*" -and $worker.CommandLine -like "*$expectedModule*") {
                            Stop-Process -Id $worker.ProcessId -Force -ErrorAction SilentlyContinue
                        }
                    }
                    Stop-Process -Id $savedPid -Force -ErrorAction SilentlyContinue
                    $stopped = $true
                } else {
                    Write-Host "Ignoring stale PID $savedPid for $name (reused by another process)."
                }
            }
        }
        Remove-Item -LiteralPath $pidFile -ErrorAction SilentlyContinue
    }

    # Fallback: stop any remaining project service listening on the port
    $listeners = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($listeners) {
        foreach ($ownerPid in ($listeners.OwningProcess | Select-Object -Unique)) {
            $ownerProc = Get-CimInstance Win32_Process -Filter "ProcessId = $ownerPid" -ErrorAction SilentlyContinue
            if ($ownerProc -and $ownerProc.Name -like "python*" -and $ownerProc.CommandLine -like "*$expectedModule*") {
                Stop-Process -Id $ownerPid -Force -ErrorAction SilentlyContinue
                $stopped = $true
            }
        }
    }

    if ($stopped) {
        Write-Host "Stopped $name."
    } else {
        Write-Host "$name was not running."
    }
}
