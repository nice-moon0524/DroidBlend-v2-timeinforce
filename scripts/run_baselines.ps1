param([Parameter(ValueFromRemainingArguments=$true)][string[]]$RemainingArgs)
$out = $(if ($env:DROIDBLEND_OUTPUT_DIR) { $env:DROIDBLEND_OUTPUT_DIR } else { "results/hybrid_baselines" })
& "$PSScriptRoot\run_all.ps1" -Mode baselines -OutputDir $out @RemainingArgs

