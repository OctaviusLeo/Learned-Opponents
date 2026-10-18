# Build the LearnedOpponentsEditor target (task 15.1):
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build_game.ps1
# Writes runs\logs\build_game.log and runs\logs\build_game.exit (exit code). A cold build can
# take a while; start it in the background and poll the .exit file. Extra arguments go to
# UnrealBuildTool (for example -MaxParallelActions=8 while a training run uses the CPU).
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$UbtArgs)
. "$PSScriptRoot\ue_common.ps1"
$Log = Join-Path $Logs "build_game.log"
$Exit = Join-Path $Logs "build_game.exit"
Remove-Item $Exit -ErrorAction SilentlyContinue
Wait-NoUnreal
& $BuildBat LearnedOpponentsEditor Win64 Development "-Project=$Project" -WaitMutex -NoHotReload @UbtArgs *>&1 |
    Tee-Object -FilePath $Log
$code = $LASTEXITCODE
Set-Content -Path $Exit -Value $code
Write-Host "build exit code $code"
exit $code
