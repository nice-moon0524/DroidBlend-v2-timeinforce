param(
    [string]$PythonExe = $(if ($env:DROIDBLEND_PYTHON) { $env:DROIDBLEND_PYTHON } else { "python" }),
    [string]$Sender = $(if ($env:DROIDBLEND_SENDER) { $env:DROIDBLEND_SENDER } else { "/opt/hhy/models/Mistral-7B-v0.1" }),
    [string]$Receiver = $(if ($env:DROIDBLEND_RECEIVER) { $env:DROIDBLEND_RECEIVER } else { "/opt/hhy/models/mistrallite" }),
    [string]$Device = $(if ($env:DROIDBLEND_DEVICE) { $env:DROIDBLEND_DEVICE } else { "cuda" }),
    [string]$SenderDevice = $(if ($env:DROIDBLEND_SENDER_DEVICE) { $env:DROIDBLEND_SENDER_DEVICE } else { "" }),
    [string]$ReceiverDevice = $(if ($env:DROIDBLEND_RECEIVER_DEVICE) { $env:DROIDBLEND_RECEIVER_DEVICE } else { "" }),
    [string]$DType = $(if ($env:DROIDBLEND_DTYPE) { $env:DROIDBLEND_DTYPE } else { "auto" }),
    [string]$DataDir = $(if ($env:DROIDBLEND_DATA_DIR) { $env:DROIDBLEND_DATA_DIR } else { "data/processed" }),
    [string]$Profile = $(if ($env:DROIDBLEND_PROFILE) { $env:DROIDBLEND_PROFILE } else { "../DroidSpeak-new/profiling_results.json" }),
    [string]$OutputDir = $(if ($env:DROIDBLEND_OUTPUT_DIR) { $env:DROIDBLEND_OUTPUT_DIR } else { "results/hybrid" }),
    [int]$MaxSamples = $(if ($env:DROIDBLEND_MAX_SAMPLES) { [int]$env:DROIDBLEND_MAX_SAMPLES } else { 0 }),
    [int]$MaxNewTokens = $(if ($env:DROIDBLEND_MAX_NEW_TOKENS) { [int]$env:DROIDBLEND_MAX_NEW_TOKENS } else { 20 }),
    [int]$MaxPromptTokens = $(if ($env:DROIDBLEND_MAX_PROMPT_TOKENS) { [int]$env:DROIDBLEND_MAX_PROMPT_TOKENS } else { 0 }),
    [ValidateSet("baselines", "sweep", "all")][string]$Mode = "all",
    [string]$RecomputeSpans = $(if ($env:DROIDBLEND_RECOMPUTE_SPANS) { $env:DROIDBLEND_RECOMPUTE_SPANS } else { "pareto" }),
    [string]$TokenRatios = $(if ($env:DROIDBLEND_TOKEN_RATIOS) { $env:DROIDBLEND_TOKEN_RATIOS } else { "0.10,0.20,0.30,0.40,0.50" }),
    [float]$TokenRatio = $(if ($env:DROIDBLEND_TOKEN_RATIO) { [float]$env:DROIDBLEND_TOKEN_RATIO } else { 0.15 }),
    [float]$MaxF1Drop = $(if ($env:DROIDBLEND_MAX_F1_DROP) { [float]$env:DROIDBLEND_MAX_F1_DROP } else { 0.0 }),
    [switch]$LocalFilesOnly,
    [switch]$TrustRemoteCode,
    [switch]$NoFullPrefillBaseline
)

$ErrorActionPreference = "Stop"
$experimentArgs = @(
    "-m", "experiments.run_hybrid_experiment",
    "--sender", $Sender,
    "--receiver", $Receiver,
    "--profiling-results", $Profile,
    "--data-dir", $DataDir,
    "--output-dir", $OutputDir,
    "--device", $Device,
    "--dtype", $DType,
    "--mode", $Mode,
    "--max-new-tokens", $MaxNewTokens.ToString(),
    "--token-recompute-ratio", $TokenRatio.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "--recompute-spans", $RecomputeSpans,
    "--token-ratios", $TokenRatios
)
if ($SenderDevice) { $experimentArgs += @("--sender-device", $SenderDevice) }
if ($ReceiverDevice) { $experimentArgs += @("--receiver-device", $ReceiverDevice) }
if ($MaxSamples -gt 0) { $experimentArgs += @("--max-samples", $MaxSamples.ToString()) }
if ($MaxPromptTokens -gt 0) { $experimentArgs += @("--max-prompt-tokens", $MaxPromptTokens.ToString()) }
if ($LocalFilesOnly -or $env:DROIDBLEND_LOCAL_FILES_ONLY -eq "1") { $experimentArgs += "--local-files-only" }
if ($TrustRemoteCode -or $env:DROIDBLEND_TRUST_REMOTE_CODE -eq "1") { $experimentArgs += "--trust-remote-code" }
if ($NoFullPrefillBaseline) { $experimentArgs += "--no-full-prefill-baseline" }

& $PythonExe @experimentArgs
if ($LASTEXITCODE -ne 0) { throw "DroidBlend experiment failed with exit code $LASTEXITCODE" }

& $PythonExe -m scripts.select_hybrid_config --summary "$OutputDir/summary.json" --output "$OutputDir/selected_config.json" --max-f1-drop $MaxF1Drop.ToString([System.Globalization.CultureInfo]::InvariantCulture)
if ($LASTEXITCODE -ne 0) { throw "DroidBlend selection failed with exit code $LASTEXITCODE" }

& $PythonExe -m scripts.plot_hybrid_results --summary "$OutputDir/summary.json" --output-dir "$OutputDir/figures"
if ($LASTEXITCODE -ne 0) { throw "DroidBlend plotting failed with exit code $LASTEXITCODE" }


