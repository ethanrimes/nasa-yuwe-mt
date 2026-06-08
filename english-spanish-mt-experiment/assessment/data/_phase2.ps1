cd Q:\english-spanish-mt-experiment
$ErrorActionPreference = "Continue"
$log = "assessment\data\run2.log"
"PHASE2 START $(Get-Date -Format o)" | Out-File $log -Encoding utf8
function Step($name, $argline) {
  "`n===== $name  $(Get-Date -Format o) =====" | Tee-Object -FilePath $log -Append
  uv run python @argline 2>&1 | Tee-Object -FilePath $log -Append
  "----- $name exit=$LASTEXITCODE -----" | Tee-Object -FilePath $log -Append
}
Step "GEN_INDOMAIN" @("assessment\run_generation.py","--selection","indomain","--n","60")
Step "ATTRIB"       @("assessment\error_attribution.py")
Step "METRIC"       @("assessment\metric_quality.py")
Step "REPORT"       @("assessment\make_report.py")
"`nPHASE2 DONE $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
