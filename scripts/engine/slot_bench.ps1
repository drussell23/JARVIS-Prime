<#
.SYNOPSIS
  Drive a deterministic slot bench (jarvis scripts/soak/slot_bench.py) to the
  end, resuming across VM stops and reboots.

.DESCRIPTION
  Replaces slot_arms.ps1. That driver kept no progress and its arms took
  whatever the live sensors produced, so the 2026-10-10 A/B compared
  different work, and when the WSL VM stopped under arm 2 the run was gone.

  This driver holds no state of its own. Each pass it asks the bench which
  arm is current, sets J-Prime to that arm's slot count, and runs the arm in
  the FOREGROUND (its wsl.exe is what keeps the VM alive). Whatever ends the
  arm -- completion, a signal, the VM stopping -- the next pass reads
  state.json again and resumes. -Install registers it as the OV-SlotBench
  task (runs now, and again at logon after a reboot); -Stop ends it cleanly.

.EXAMPLE
  slot_bench.ps1 -Run sb-20261013 -Install
  slot_bench.ps1 -Run sb-20261013 -Stop
#>
param(
  [Parameter(Mandatory = $true)][string]$Run,
  [string]$Preload = 'qwen3-coder-ov:30b',
  [int]$Ctx = 32768,
  [string]$Restore = 'auto',
  [string]$Bench = '/home/jarvis_svc/jarvis/scripts/soak/slot_bench.py',
  [string]$Python = '/home/jarvis_svc/.venvs/ov/bin/python3',
  [string]$TaskName = 'OV-SlotBench',
  [switch]$Install,
  [switch]$Stop
)
$ErrorActionPreference = 'Stop'
$engine = Join-Path $PSScriptRoot 'jprime_engine.ps1'
$stateDir = Join-Path $env:LOCALAPPDATA 'JARVIS\jprime\state'
$log = Join-Path $stateDir 'slot_bench.log'
$stopMarker = Join-Path $stateDir "slot_bench-$Run.stop"
$wsl = Join-Path $env:WINDIR 'System32\wsl.exe'
New-Item -ItemType Directory -Force -Path $stateDir | Out-Null
function Say($m) { "$(Get-Date -Format s) [$Run] $m" | Add-Content -Path $log }
function Bench([string[]]$a) { & $wsl -d Ubuntu -u jarvis_svc -- $Python $Bench @a }

if ($Stop) {
  New-Item -ItemType File -Force -Path $stopMarker | Out-Null
  & $wsl -d Ubuntu -u jarvis_svc -- pkill -TERM -f "slot_bench.py arm --run $Run"
  Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
  Say 'stop requested: arm signalled (it checkpoints and stops its soak), resume task removed'
  return
}

if ($Install) {
  Remove-Item -Force -ErrorAction SilentlyContinue $stopMarker
  $args_ = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$PSCommandPath`" -Run $Run"
  $action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $args_
  $trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
  # No time limit: the bench ends itself; a limit would kill the VM's client.
  $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
  Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Force | Out-Null
  Start-ScheduledTask -TaskName $TaskName
  Say "installed $TaskName (runs now, resumes at logon)"
  return
}

function EngineSlots {
  try {
    $m = (Invoke-RestMethod http://127.0.0.1:8000/api/ps -TimeoutSec 5).models |
      Where-Object { $_.name -eq $Preload }
    if ($m) { return [int]$m.parallel }
  } catch {}
  return 0
}

function EnsureEngine([int]$n) {
  if ((EngineSlots) -eq $n) { return }
  & $engine -Action stop | Out-Null
  Start-Sleep -Seconds 5
  & $engine -Action start -Preload $Preload -Ctx $Ctx -Parallel $n | Out-Null
  $deadline = (Get-Date).AddMinutes(10)
  while ((Get-Date) -lt $deadline) {
    if ((EngineSlots) -eq $n) { Say "engine serving $Preload with $n slot(s)"; return }
    Start-Sleep -Seconds 5
  }
  throw "engine did not come up with $n slot(s)"
}

Say 'driver start'
$failures = 0
while (-not (Test-Path $stopMarker)) {
  $raw = Bench @('status', '--run', $Run) | Select-Object -Last 1
  try { $st = $raw | ConvertFrom-Json } catch { $st = $null }
  if ($null -eq $st) {
    # The VM may be restarting under us; that is what resume is for.
    $failures++; Say "status unreadable ($raw); retry $failures"
    if ($failures -gt 10) { Say 'giving up: bench status unreadable'; exit 1 }
    Start-Sleep -Seconds 60; continue
  }
  if ($st.status -eq 'all_done') { break }
  try { EnsureEngine ([int]$st.slots) } catch { $failures++; Say "$_"; Start-Sleep -Seconds 60; continue }
  Say "arm $($st.current) slots=$($st.slots) status=$($st.status) active=$($st.active_s)s"
  Bench @('arm', '--run', $Run) | Out-Null
  $rc = $LASTEXITCODE
  Say "arm pass ended rc=$rc"
  if ($rc -eq 0 -or $rc -eq 10) { $failures = 0; continue }
  # 75 = interrupted (signal); anything else = driver error or the VM went
  # away under the wsl.exe. Either way state.json holds the progress.
  $failures++
  if ($failures -gt 10) { Say 'giving up after 10 failed passes'; exit 1 }
  Start-Sleep -Seconds 60
}

if (Test-Path $stopMarker) { Say 'stopped by request'; return }
Bench @('report', '--run', $Run) | Out-File -Encoding utf8 (Join-Path $stateDir "slot_bench-$Run-report.json")
& $engine -Action stop | Out-Null
Start-Sleep -Seconds 5
& $engine -Action start -Preload $Preload -Ctx $Ctx -Parallel $Restore | Out-Null
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
Say "all arms done; report saved; engine restored to $Restore"
