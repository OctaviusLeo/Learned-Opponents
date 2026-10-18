# Record the README gameplay GIF (task 16.2):
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\capture_gif.ps1
# Launches RaceLevel windowed with the ONNX opponent and the scripted demo player (-LODemoPlayer),
# waits until the race is running, records 10 s of the window with FFmpeg gdigrab, and writes
# docs\media\gameplay.gif at 640 px wide (must stay under 10 MB).
param([int]$Seconds = 10, [string]$Model = "")
. "$PSScriptRoot\ue_common.ps1"
$Ffmpeg = (Get-Command ffmpeg).Source
$Media = Join-Path $Repo "docs\media"
$Raw = Join-Path $Logs "gameplay_raw.mp4"
$Gif = Join-Path $Media "gameplay.gif"
$GameLog = Join-Path $Logs "race_game.log"
New-Item -ItemType Directory -Force $Media | Out-Null
Remove-Item $Raw, $GameLog -ErrorAction SilentlyContinue

Wait-NoUnreal
if (Test-GpuJobRunning) { throw "a GPU job is running; refusing to start the game" }
$gameArgs = @("`"$Project`"", "/Game/Maps/RaceLevel", "-game", "-windowed", "-ResX=1280", "-ResY=720", "-LODemoPlayer",
          "-LOQuitAfter=600", "-nosplash", "-NoSound", "-abslog=`"$GameLog`"")
if ($Model) { $gameArgs += "-LOModel=`"$Model`"" }
$game = Start-Process -FilePath $Editor -ArgumentList $gameArgs -PassThru
try {
    # Wait for the race manager to report ready (first launch can spend minutes compiling shaders).
    $deadline = (Get-Date).AddMinutes(20)
    while (-not ((Test-Path $GameLog) -and (Select-String -Path $GameLog -Pattern "race ready" -Quiet))) {
        if ((Get-Date) -gt $deadline -or $game.HasExited) { throw "the race did not start; see $GameLog" }
        Start-Sleep -Seconds 2
    }
    Start-Sleep -Seconds 4
    $game.Refresh()
    $title = $game.MainWindowTitle
    if (-not $title) { throw "no game window title found" }
    Write-Host "recording window '$title' for $Seconds s"
    & $Ffmpeg -y -loglevel error -f gdigrab -framerate 20 -t $Seconds -i "title=$title" -c:v libx264 -pix_fmt yuv420p $Raw
    if ($LASTEXITCODE -ne 0) { throw "ffmpeg capture failed" }
} finally {
    if (-not $game.HasExited) { Stop-Process -Id $game.Id -Force }
}
& $Ffmpeg -y -loglevel error -i $Raw -vf "fps=15,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer" $Gif
if ($LASTEXITCODE -ne 0) { throw "ffmpeg GIF conversion failed" }
$mb = (Get-Item $Gif).Length / 1MB
Write-Host ("wrote {0} ({1:N2} MB)" -f $Gif, $mb)
Select-String -Path $GameLog -Pattern "race ready" | ForEach-Object { $_.Line }
if ($mb -ge 10) { throw "GIF is $mb MB, over the 10 MB limit" }
