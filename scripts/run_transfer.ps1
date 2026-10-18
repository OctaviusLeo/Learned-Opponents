# Transfer check, one command (task 17.1):
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_transfer.ps1
# 1. Python writes the 50 shared layouts (fixed seed) to results\transfer\episodes.json.
# 2. MuJoCo plays all six policies on them (WSL, GPU).
# 3. The game plays the same policies on the same layouts headless (TransferArena).
# 4. Python pairs the results and writes the transfer table into results\summary.md.
. "$PSScriptRoot\ue_common.ps1"
function Invoke-Wsl([string]$Cmd) {
    wsl -d Ubuntu-24.04 -- bash -c "cd /mnt/c/dev/Projects/Learned-Opponents && $Cmd"
    if ($LASTEXITCODE -ne 0) { throw "WSL step failed: $Cmd" }
}
$Py = "~/.venvs/learned-opponents/bin/python"
Invoke-Wsl "$Py -m learned_opponents.eval.transfer episodes"
Invoke-Wsl "$Py -m learned_opponents.eval.transfer mujoco"

Wait-NoUnreal
if (Test-GpuJobRunning) { throw "a GPU job is running; refusing to start the game" }
$Log = Join-Path $Logs "transfer_game.log"
$Out = Join-Path $Repo "results\transfer\game_results.json"
Remove-Item $Out -ErrorAction SilentlyContinue
& $EditorCmd $Project "/Game/Maps/TransferArena" -game -nullrhi -unattended -nopause -nosplash -NoSound -log `
    "-LOOut=$Out" *>&1 | Out-File -FilePath $Log -Encoding utf8
if (-not (Test-Path $Out)) { throw "the game wrote no results; see $Log" }
Select-String -Path $Log -Pattern "transfer:" | ForEach-Object { $_.Line }

Invoke-Wsl "$Py -m learned_opponents.eval.transfer summarize"
