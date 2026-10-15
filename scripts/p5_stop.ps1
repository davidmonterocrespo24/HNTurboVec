# Stops the P5 workers (scripts/p5_build.py lm and embed) and the tools they are running. The month
# in progress is built again on the next start (scripts\p5_start.ps1, or at logon).
#   powershell -ExecutionPolicy Bypass -File scripts\p5_stop.ps1
$ws = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'p5_build\.py\s+(lm|embed)' })
foreach ($w in $ws) {
    & taskkill.exe /PID $w.ProcessId /T /F | Out-Null
}
"stopped $($ws.Count) worker process(es)"
