# End-to-end gate (task 19):
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\verify_e2e.ps1
# Passes only when every check exits 0: the full pytest suite (WSL), the Unreal automation tests,
# parity on all six models, a green CI run on the latest pushed commit, the nominal / robustness /
# transfer sections in results\summary.md, the gameplay GIF, and README blocks matching the summary.
. "$PSScriptRoot\ue_common.ps1"
$ErrorActionPreference = "Continue"
$results = [ordered]@{}
function Step([string]$Name, [scriptblock]$Body) {
    Write-Host "== $Name"
    & $Body
    $ok = ($LASTEXITCODE -eq 0)
    $results[$Name] = $ok
    Write-Host ("{0}  {1}" -f ($(if ($ok) { "PASS" } else { "FAIL" })), $Name)
}
function Wsl([string]$Cmd) { wsl -d Ubuntu-24.04 -- bash -c "cd /mnt/c/dev/Projects/Learned-Opponents && $Cmd" }
$Py = "~/.venvs/learned-opponents/bin/python"

Step "full pytest (WSL)" { Wsl "$Py -m pytest -q -p no:cacheprovider tests" }
Step "Unreal automation tests" { & "$PSScriptRoot\run_ue_tests.ps1" }
Step "parity, six models" { Wsl "$Py -m learned_opponents.export.parity" }
Step "summary sections" {
    $s = Get-Content (Join-Path $Repo "results\summary.md") -Raw -Encoding utf8
    $need = @("## Nominal evaluation: baseline", "## Robustness grid", "## Transfer check")
    $missing = $need | Where-Object { -not $s.Contains($_) }
    if ($missing) { Write-Host "missing: $missing"; $global:LASTEXITCODE = 1 } else { $global:LASTEXITCODE = 0 }
}
Step "gameplay GIF" {
    $g = Join-Path $Repo "docs\media\gameplay.gif"
    if ((Test-Path $g) -and ((Get-Item $g).Length -lt 10MB)) { $global:LASTEXITCODE = 0 } else { $global:LASTEXITCODE = 1 }
}
Step "README matches summary.md" { Wsl "$Py -m learned_opponents.eval.readme --check" }
Step "CI green on the latest pushed commit" {
    $head = (git -C $Repo rev-parse HEAD).Trim()
    $remote = (git -C $Repo rev-parse origin/master).Trim()
    $run = gh run list --repo OctaviusLeo/Learned-Opponents --commit $head --limit 1 --json conclusion,status,headSha | ConvertFrom-Json
    Write-Host "HEAD $head, origin/master $remote, run: $($run.status) $($run.conclusion)"
    if ($head -eq $remote -and $run -and $run[0].conclusion -eq "success") { $global:LASTEXITCODE = 0 } else { $global:LASTEXITCODE = 1 }
}
Write-Host ""
$results.GetEnumerator() | ForEach-Object { Write-Host ("{0}  {1}" -f ($(if ($_.Value) { "PASS" } else { "FAIL" })), $_.Key) }
if ($results.Values -contains $false) { Write-Host "end-to-end gate: FAIL"; exit 1 }
Write-Host "end-to-end gate: PASS"
exit 0
