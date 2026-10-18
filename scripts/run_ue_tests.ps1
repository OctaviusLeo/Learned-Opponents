# Run every Unreal automation test under the "LearnedOpponents." prefix headless (tasks 15-16):
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_ue_tests.ps1
# Exits 0 only when at least one test ran and none failed. The report goes to runs\ue_tests\.
# -Filter narrows the run (default: every test under "LearnedOpponents").
param([string]$Filter = "LearnedOpponents")
. "$PSScriptRoot\ue_common.ps1"
$Report = Join-Path $Repo "runs\ue_tests"
Remove-Item -Recurse -Force $Report -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $Report | Out-Null
$Log = Join-Path $Logs "run_ue_tests.log"
foreach ($f in @("results\game\task.json", "results\fixtures\obs_cases.json")) {
    if (-not (Test-Path (Join-Path $Repo $f))) { Write-Host "missing $f (run scripts/make_obs_fixtures.sh in WSL)" }
}
Wait-NoUnreal
& $EditorCmd $Project "-ExecCmds=Automation RunTests $Filter;Quit" -unattended -nopause -nosplash -nullrhi `
    "-testexit=Automation Test Queue Empty" "-ReportExportPath=$Report" -log -NoSound *>&1 | Out-File -FilePath $Log -Encoding utf8
$index = Join-Path $Report "index.json"
if (-not (Test-Path $index)) {
    Write-Host "no test report at $index; see $Log"
    exit 1
}
$r = Get-Content $index -Raw -Encoding utf8 | ConvertFrom-Json
$passed = [int]$r.succeeded + [int]$r.succeededWithWarnings
Write-Host ("tests: {0} passed, {1} failed, {2} not run" -f $passed, $r.failed, $r.notRun)
foreach ($t in $r.tests) {
    Write-Host ("  {0,-9} {1}" -f $t.state, $t.fullTestPath)
    foreach ($e in $t.entries) {
        if ($e.event.type -ne "Info" -or $t.state -ne "Success") { Write-Host ("            {0}: {1}" -f $e.event.type, $e.event.message) }
        elseif ($e.event.message -match "max \|") { Write-Host ("            {0}" -f $e.event.message) }
    }
}
if ($passed -ge 1 -and [int]$r.failed -eq 0) { exit 0 } else { exit 1 }
