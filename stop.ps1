$ErrorActionPreference = "Stop"
$runtimePath = Join-Path $PSScriptRoot ".runtime"
foreach ($name in @("ui", "api")) {
    $pidFile = Join-Path $runtimePath "$name.pid"
    if (-not (Test-Path -LiteralPath $pidFile)) { continue }
    $savedPid = [int](Get-Content -LiteralPath $pidFile)
    $serviceProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $savedPid" -ErrorAction SilentlyContinue
    if ($serviceProcess) {
        $expectedPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
        $expectedModule = if ($name -eq "api") { "uvicorn app.api:app" } else { "streamlit run app/main.py" }
        if ($serviceProcess.ExecutablePath -ne $expectedPython -or $serviceProcess.CommandLine -notlike "*$expectedModule*") {
            throw "PID $savedPid does not match this project's $name service. It was not stopped."
        }
        # A Windows venv launcher may own the actual Python worker as a child.
        $workers = Get-CimInstance Win32_Process -Filter "ParentProcessId = $savedPid"
        foreach ($worker in $workers) {
            if ($worker.Name -eq "python.exe" -and $worker.CommandLine -like "*$expectedModule*") {
                Stop-Process -Id $worker.ProcessId
            }
        }
        Stop-Process -Id $savedPid -ErrorAction SilentlyContinue
        Write-Host "Stopped $name."
    }
    Remove-Item -LiteralPath $pidFile
}
