# Record the README gameplay GIF (task 16.2):
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\capture_gif.ps1
# Launches RaceLevel windowed with the ONNX opponent and the scripted demo player (-LODemoPlayer).
# The game itself saves its viewport (with the HUD) at 15 fps for 10 s of race time
# (-LOCaptureDir), so the capture does not depend on window position or display scaling; FFmpeg then
# writes docs\media\gameplay.gif at 640 px wide (must stay under 10 MB).
param([int]$Seconds = 10, [int]$Fps = 15, [double]$Start = 3, [string]$Model = "")
. "$PSScriptRoot\ue_common.ps1"
$Ffmpeg = (Get-Command ffmpeg).Source
$Media = Join-Path $Repo "docs\media"
$Frames = Join-Path $Repo "runs\media\frames"
$Gif = Join-Path $Media "gameplay.gif"
$GameLog = Join-Path $Logs "race_game.log"
New-Item -ItemType Directory -Force $Media | Out-Null
Remove-Item -Recurse -Force $Frames -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $Frames | Out-Null
Remove-Item $GameLog -ErrorAction SilentlyContinue

Wait-NoUnreal
if (Test-GpuJobRunning) { throw "a GPU job is running; refusing to start the game" }
$quit = $Start + $Seconds + 1.5
$gameArgs = @("`"$Project`"", "/Game/Maps/RaceLevel", "-game", "-windowed", "-ResX=1280", "-ResY=720", "-LODemoPlayer",
          "-LOCaptureDir=`"$Frames`"", "-LOCaptureStart=$Start", "-LOCaptureSeconds=$Seconds", "-LOCaptureFps=$Fps",
          "-LOQuitAfter=$quit", "-nosplash", "-NoSound", "-abslog=`"$GameLog`"")
if ($Model) { $gameArgs += "-LOModel=`"$Model`"" }
$game = Start-Process -FilePath $Editor -ArgumentList $gameArgs -PassThru
if (-not $game.WaitForExit(20 * 60 * 1000)) {        # the first launch can spend minutes compiling shaders
    Stop-Process -Id $game.Id -Force
    throw "the race did not finish within 20 minutes; see $GameLog"
}
$n = (Get-ChildItem $Frames -Filter "frame_*.png").Count
Write-Host "game exited; $n frames in $Frames"
if ($n -lt $Seconds * $Fps * 0.9) { throw "only $n frames captured; see $GameLog" }
& $Ffmpeg -y -loglevel error -framerate $Fps -i (Join-Path $Frames "frame_%04d.png") `
    -vf "scale=640:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer" $Gif
if ($LASTEXITCODE -ne 0) { throw "ffmpeg GIF conversion failed" }
$mb = (Get-Item $Gif).Length / 1MB
Write-Host ("wrote {0} ({1:N2} MB, {2} frames at {3} fps)" -f $Gif, $mb, $n, $Fps)
Select-String -Path $GameLog -Pattern "race ready|race: " | ForEach-Object { $_.Line.Substring($_.Line.IndexOf("race")) }
if ($mb -ge 10) { throw "GIF is $mb MB, over the 10 MB limit" }
