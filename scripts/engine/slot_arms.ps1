<#
.SYNOPSIS
  Run O+V soak arms back to back, one J-Prime slot count per arm.

.DESCRIPTION
  For each count in -Slots: wait until no soak supervisor runs in WSL, restart
  the engine with -Parallel <count> (and the preload), then start a soak of
  -ArmSeconds through jarvis' start_detached_soak.ps1. After the last arm the
  engine returns to -Restore. Each arm's session id and slot count are
  appended to %LOCALAPPDATA%\JARVIS\jprime\state\slot_arms.log.

.EXAMPLE
  slot_arms.ps1 -Slots 2,1 -ArmSeconds 7200
#>
param(
  # A comma list as ONE string ("2,1"): under `powershell -File` an [int[]]
  # parameter receives "2,1" as a single string and casts it to 21.
  [string]$Slots = '2,1',
  [int]$ArmSeconds = 7200,
  [string]$Preload = 'qwen3-coder-ov:30b',
  [int]$Ctx = 32768,
  [string]$Restore = 'auto',
  [string]$SoakLauncher = 'C:\Users\Jarvis\Desktop\TrinityAi\jarvis\scripts\windows\start_detached_soak.ps1'
)
$ErrorActionPreference = 'Stop'
$engine = Join-Path $PSScriptRoot 'jprime_engine.ps1'
$log = Join-Path $env:LOCALAPPDATA 'JARVIS\jprime\state\slot_arms.log'
$wsl = Join-Path $env:WINDIR 'System32\wsl.exe'
function Say($m) { "$(Get-Date -Format s) $m" | Add-Content -Path $log }
function SoakRunning { & $wsl -d Ubuntu -u jarvis_svc -- pgrep -f 'battle_test.terminal_supervisor' | Out-Null; return ($LASTEXITCODE -eq 0) }
function WaitSoakDone { while (SoakRunning) { Start-Sleep -Seconds 60 } }

$counts = @($Slots -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ } | ForEach-Object {
  $v = 0
  if (-not [int]::TryParse($_, [ref]$v) -or $v -lt 1) { Say "bad slot count '$_'; stopping"; exit 2 }
  $v })
Say "driver start: slots=$($counts -join ',') arm=${ArmSeconds}s"
foreach ($n in $counts) {
  WaitSoakDone
  & $engine -Action stop | Out-Null
  Start-Sleep -Seconds 5
  & $engine -Action start -Preload $Preload -Ctx $Ctx -Parallel $n | Out-Null
  $deadline = (Get-Date).AddMinutes(10); $ps = $null
  while ((Get-Date) -lt $deadline) {
    try { $ps = (Invoke-RestMethod http://127.0.0.1:8000/api/ps -TimeoutSec 5).models | Where-Object { $_.name -eq $Preload } } catch {}
    if ($ps) { break }; Start-Sleep -Seconds 5
  }
  if (-not $ps) { Say "slots=$n engine did not load $Preload; stopping"; exit 1 }
  Say "slots=$($ps.parallel) vram=$([int]($ps.size_vram / 1MB))MiB -> soak ${ArmSeconds}s"
  & $SoakLauncher -MaxWallSeconds $ArmSeconds | ForEach-Object { Say "  $_" }
  Start-Sleep -Seconds 120
}
WaitSoakDone
& $engine -Action stop | Out-Null
Start-Sleep -Seconds 5
& $engine -Action start -Preload $Preload -Ctx $Ctx -Parallel $Restore | Out-Null
Say "driver done: engine restored to $Restore"
