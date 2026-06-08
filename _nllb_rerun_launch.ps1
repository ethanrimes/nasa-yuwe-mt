$ErrorActionPreference = 'Continue'
Set-Location 'Q:\nasa-yuwe-mt\english-chinese-mt-experiment'
$env:H100_USE_SPOT = '0'
$env:WANDB_API_KEY = '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'
& 'Q:\nasa-yuwe-mt\.venv\Scripts\python.exe' scripts\11_run_ablations.py run --only-nllb --yes --max-budget-hours 8 *>&1 |
    Tee-Object -FilePath 'Q:\nasa-yuwe-mt\_nllb_rerun2_20260608_113148.log'
