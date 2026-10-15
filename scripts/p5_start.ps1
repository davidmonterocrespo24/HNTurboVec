# Starts (or resumes) the two P5 workers that are not running: scripts/p5_build.py lm and embed,
# detached from any terminal (the embed one lowers its own priority). Each worker skips the months
# it has finished, so starting again after a shutdown loses at most the month in progress.
# Runs at logon from the Startup folder (hnz-p5.cmd); by hand:
#   powershell -ExecutionPolicy Bypass -File scripts\p5_start.ps1
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$log = Join-Path $root 'logs\p5_start.log'
function Say($m) { Add-Content -Path $log -Value ("{0:yyyy-MM-dd HH:mm:ss} {1}" -f (Get-Date), $m) }
$running = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'p5_build\.py\s+(lm|embed)' })
foreach ($w in 'lm', 'embed') {
    if ($running | Where-Object { $_.CommandLine -match "p5_build\.py\s+$w(\s|$)" }) {
        Say "${w}: already running"
        continue
    }
    $p = Start-Process -FilePath (Join-Path $root '.venv\Scripts\python.exe') -ArgumentList 'scripts\p5_build.py', $w `
        -WorkingDirectory $root -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $root "logs\p5_$w.out") -RedirectStandardError (Join-Path $root "logs\p5_$w.err")
    Say "${w}: started (pid $($p.Id))"
}
