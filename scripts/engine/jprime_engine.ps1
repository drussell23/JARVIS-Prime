<#
.SYNOPSIS
  Start, stop or query the J-Prime Engine on a Windows GPU host.

.DESCRIPTION
  Idempotent: `start` returns at once if an engine is already serving on the
  port, otherwise launches `python -m jarvis_prime.engine` detached (it
  outlives the caller, including a WSL shell that invoked this through
  interop) and waits until the HTTP surface answers. The model preload then
  continues in the background; /api/ps shows when it is resident.

  The engine runs from its own venv (prime\.venv-engine) holding only the
  engine's five dependencies, created on first use with uv. llama-server is
  a child of the engine bound by a kill-on-close Job Object, so stopping the
  engine always frees the GPU.

.EXAMPLE
  jprime_engine.ps1 -Action start -Preload qwen3-coder-ov:30b
  jprime_engine.ps1 -Action status
  jprime_engine.ps1 -Action stop
#>
param(
  [ValidateSet('start', 'stop', 'status')] [string]$Action = 'status',
  [int]$Port = 8000,
  [string]$Preload = '',
  [int]$Ctx = 32768,
  [int]$WaitSeconds = 60
)
$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$State = Join-Path $env:LOCALAPPDATA 'JARVIS\jprime\state'
$PidFile = Join-Path $State "engine-$Port.pid"
$Venv = Join-Path $Repo '.venv-engine'
$Py = Join-Path $Venv 'Scripts\python.exe'
New-Item -ItemType Directory -Force $State | Out-Null

function Test-Serving {
  try { Invoke-RestMethod "http://127.0.0.1:$Port/api/version" -TimeoutSec 2 | Out-Null; return $true } catch { return $false }
}

function Get-EnginePid {
  if (-not (Test-Path $PidFile)) { return $null }
  $p = [int](Get-Content $PidFile -Raw)
  if (Get-Process -Id $p -ErrorAction SilentlyContinue) { return $p }
  return $null
}

switch ($Action) {
  'status' {
    if (Test-Serving) {
      $h = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 5
      "serving on :$Port pid=$(Get-EnginePid) resident=" + (($h.resident | ForEach-Object { $_.name }) -join ',')
      exit 0
    }
    "not serving on :$Port"; exit 3
  }
  'stop' {
    $p = Get-EnginePid
    if ($p) { Stop-Process -Id $p -Force; Remove-Item $PidFile -ErrorAction SilentlyContinue; "stopped pid $p" }
    else { "no engine recorded on :$Port" }
    exit 0
  }
  'start' {
    if (Test-Serving) { "already serving on :$Port"; exit 0 }
    if (-not (Test-Path $Py)) {
      $uv = Join-Path $env:USERPROFILE '.local\bin\uv.exe'
      & $uv venv --python 3.11 $Venv | Out-Null
      & $uv pip install --python $Py fastapi uvicorn httpx pyyaml psutil | Out-Null
    }
    $log = Join-Path $State "engine-$Port.log"
    $cmd = "`"$Py`" -m jarvis_prime.engine --port $Port --ctx $Ctx --log-file `"$log`""
    if ($Preload) { $cmd += " --preload $Preload" }
    # Win32_Process.Create, not Start-Process: the engine must share NO handles
    # with this shell. Start-Process with redirection creates the child with
    # handle inheritance on, so an engine launched from WSL through interop
    # kept the caller's pipe open and the caller never returned.
    $startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
      CommandLine = $cmd; CurrentDirectory = $Repo; ProcessStartupInformation = $startup }
    if ($r.ReturnValue -ne 0) { "Win32_Process.Create failed rc=$($r.ReturnValue)"; exit 1 }
    $enginePid = [int]$r.ProcessId
    Set-Content -Path $PidFile -Value $enginePid
    $deadline = (Get-Date).AddSeconds($WaitSeconds)
    while ((Get-Date) -lt $deadline) {
      if (-not (Get-Process -Id $enginePid -ErrorAction SilentlyContinue)) { "engine exited; see $log"; exit 1 }
      if (Test-Serving) { "started pid $enginePid on :$Port (preload: $(if ($Preload) { $Preload } else { 'none' }))"; exit 0 }
      Start-Sleep -Milliseconds 500
    }
    "engine did not answer within ${WaitSeconds}s; see $log"; exit 1
  }
}
