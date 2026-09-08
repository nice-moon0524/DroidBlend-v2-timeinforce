param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$RemainingArgs
)

$ErrorActionPreference = "Stop"
& "$PSScriptRoot\run_all.ps1" @RemainingArgs
