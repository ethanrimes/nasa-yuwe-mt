cd Q:\english-spanish-mt-experiment
$ErrorActionPreference = "Continue"
$log = "assessment\data\run.log"
function Step($name, $argline) {
  "`n===== $name  $(Get-Date -Format o) =====" | Tee-Object -FilePath $log -Append
  uv run python @argline 2>&1 | Tee-Object -FilePath $log -Append
  "----- $name exit=$LASTEXITCODE -----" | Tee-Object -FilePath $log -Append
}
Step "TF"        @("assessment\run_teacher_forced.py","--n","100")
Step "ENPPL"     @("assessment\run_english_ppl.py","--n","150")
Step "GEN_BEST"  @("assessment\run_generation.py","--selection","best","--n","100")
Step "GEN_TRAIL" @("assessment\run_generation.py","--selection","trail","--n","80")
"`nPIPELINE DONE $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
