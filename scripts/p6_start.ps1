# Starts (or resumes) scripts/p6_run.py if it is not running, detached from any terminal. Finished
# stages are skipped, so starting again after a shutdown repeats at most the stage in progress.
# Runs at logon from the Startup folder (hnz-p6.cmd); by hand:
#   powershell -ExecutionPolicy Bypass -File scripts\p6_start.ps1
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$log = Join-Path $root 'logs\p6_start.log'
function Say($m) { Add-Content -Path $log -Value ("{0:yyyy-MM-dd HH:mm:ss} {1}" -f (Get-Date), $m) }
$running = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'p6_run\.py' })
if ($running) { Say 'p6: already running'; exit }
$p = Start-Process -FilePath (Join-Path $root '.venv\Scripts\python.exe') -ArgumentList 'scripts\p6_run.py' `
    -WorkingDirectory $root -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $root 'logs\p6_run.out') -RedirectStandardError (Join-Path $root 'logs\p6_run.err')
Say "p6: started (pid $($p.Id))"
