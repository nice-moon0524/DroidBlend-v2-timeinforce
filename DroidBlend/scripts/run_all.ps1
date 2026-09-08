param(
    [string]$PythonExe = $(if ($env:DROIDBLEND_PYTHON) { $env:DROIDBLEND_PYTHON } else { "python" }),
    [string]$Sender = $(if ($env:DROIDBLEND_SENDER) { $env:DROIDBLEND_SENDER } else { "/opt/hhy/models/Mistral-7B-v0.1" }),
    [string]$Receiver = $(if ($env:DROIDBLEND_RECEIVER) { $env:DROIDBLEND_RECEIVER } else { "/opt/hhy/models/mistrallite" }),
    [string]$Device = $(if ($env:DROIDBLEND_DEVICE) { $env:DROIDBLEND_DEVICE } else { "cuda" }),
    [string]$DataDir = $(if ($env:DROIDBLEND_DATA_DIR) { $env:DROIDBLEND_DATA_DIR } else { "data/processed" }),
    [string]$Profile = $(if ($env:DROIDBLEND_PROFILE) { $env:DROIDBLEND_PROFILE } else { "../DroidSpeak-new/profiling_results.json" }),
    [int]$MaxSamples = $(if ($env:DROIDBLEND_MAX_SAMPLES) { [int]$env:DROIDBLEND_MAX_SAMPLES } else { 0 }),
    [int]$MaxNewTokens = $(if ($env:DROIDBLEND_MAX_NEW_TOKENS) { [int]$env:DROIDBLEND_MAX_NEW_TOKENS } else { 64 }),
    [float]$ImportantFraction = $(if ($env:DROIDBLEND_IMPORTANT_FRACTION) { [float]$env:DROIDBLEND_IMPORTANT_FRACTION } else { 0.2 }),
    [int]$TopK = $(if ($env:DROIDBLEND_TOP_K) { [int]$env:DROIDBLEND_TOP_K } else { 0 })
)

$ErrorActionPreference = "Stop"

$experimentArgs = @(
    "--sender", $Sender,
    "--receiver", $Receiver,
    "--profiling-results", $Profile,
    "--data-dir", $DataDir,
    "--device", $Device,
    "--max-new-tokens", $MaxNewTokens.ToString(),
    "--important-fraction", $ImportantFraction.ToString([System.Globalization.CultureInfo]::InvariantCulture)
)
if ($MaxSamples -gt 0) {
    $experimentArgs += @("--max-samples", $MaxSamples.ToString())
}
if ($TopK -gt 0) {
    $experimentArgs += @("--top-k", $TopK.ToString())
}

& $PythonExe -m experiments.run_droidblend_experiment @experimentArgs
if ($LASTEXITCODE -ne 0) {
    throw "DroidBlend experiment failed with exit code $LASTEXITCODE"
}

& $PythonExe -m scripts.plot_results `
    --droidblend results/droidblend_quality_latency.json `
    --token-selection results/droidblend_token_selection.json `
    --output-dir results/figures
if ($LASTEXITCODE -ne 0) {
    throw "Plotting failed with exit code $LASTEXITCODE"
}
