# Create /Game/Maps/RaceLevel and /Game/Maps/TransferArena headless (no editor UI):
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\create_maps.ps1
. "$PSScriptRoot\ue_common.ps1"
$Script = Join-Path $PSScriptRoot "unreal\create_maps.py"
$Log = Join-Path $Logs "create_maps.log"
Wait-NoUnreal
& $EditorCmd $Project -run=pythonscript "-script=$Script" -unattended -nopause -nosplash -nullrhi -NoSound *>&1 |
    Out-File -FilePath $Log -Encoding utf8
Select-String -Path $Log -Pattern "LO maps:" | ForEach-Object { $_.Line }
$maps = Join-Path $Repo "game\LearnedOpponents\Content\Maps"
$ok = (Test-Path (Join-Path $maps "RaceLevel.umap")) -and (Test-Path (Join-Path $maps "TransferArena.umap"))
Write-Host ("maps present: {0}" -f $ok)
if ($ok) { exit 0 } else { exit 1 }
