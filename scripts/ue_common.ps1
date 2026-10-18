# Shared settings for the Unreal scripts. Dot-source it: . "$PSScriptRoot\ue_common.ps1"
$ErrorActionPreference = "Stop"
$UE = "C:\Program Files (x86)\UE_5.8"
$Repo = Split-Path -Parent $PSScriptRoot
$Project = Join-Path $Repo "game\LearnedOpponents\LearnedOpponents.uproject"
$EditorCmd = Join-Path $UE "Engine\Binaries\Win64\UnrealEditor-Cmd.exe"
$Editor = Join-Path $UE "Engine\Binaries\Win64\UnrealEditor.exe"
$BuildBat = Join-Path $UE "Engine\Build\BatchFiles\Build.bat"
$Logs = Join-Path $Repo "runs\logs"
New-Item -ItemType Directory -Force $Logs | Out-Null

function Wait-NoUnreal {
    # Only one Unreal process at a time.
    while (Get-Process UnrealBuildTool, UnrealEditor* -ErrorAction SilentlyContinue) {
        Write-Host "waiting for another Unreal process to finish..."
        Start-Sleep -Seconds 10
    }
}

function Test-GpuJobRunning {
    # A project GPU job (training / evaluation / export) running inside WSL.
    $out = wsl -d Ubuntu-24.04 -- pgrep -f "learned_opponents\.(train|eval|export)" 2>$null
    return [bool]$out
}
