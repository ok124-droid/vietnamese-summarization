param(
    [ValidateSet("smoke", "validation")]
    [string]$Split = "smoke",

    [ValidateSet("overwrite", "resume")]
    [string]$Mode = "overwrite",

    [int]$BatchSize = 2,

    [string]$Device = "cpu",

    [ValidateSet("fp32", "fp16", "bf16")]
    [string]$Precision = "fp32",

    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

if ($BatchSize -lt 1) {
    throw "BatchSize must be at least 1."
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$packageRoot = Join-Path $repoRoot "scratch_transformer_inference"
$inferScript = Join-Path $packageRoot "src\infer.py"
$validatorScript = Join-Path $packageRoot "src\validate_predictions.py"
$checkpoint = Join-Path $packageRoot "checkpoints\transformer_base\best_val_loss.pt"
$tokenizer = Join-Path $packageRoot "data\tokenizer\vietnamese_spm.model"
$modelConfig = Join-Path $packageRoot "configs\transformer_summarization.yaml"

if ($Split -eq "smoke") {
    $manifest = Join-Path $repoRoot "data\test_smoke_10.jsonl"
    $outputDir = Join-Path $repoRoot "outputs\predictions\test_smoke_10"
}
else {
    $manifest = Join-Path $repoRoot "data\validation_select.jsonl"
    $outputDir = Join-Path $repoRoot "outputs\predictions\validation"
}

$requiredFiles = @(
    $inferScript,
    $validatorScript,
    $checkpoint,
    $tokenizer,
    $modelConfig,
    $manifest
)

foreach ($path in $requiredFiles) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required file not found: $path"
    }
}

if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    throw "Python command not found: $Python"
}

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

$configs = @(
    [pscustomobject]@{
        Id = "greedy"
        File = "decoding_greedy.json"
        Output = "scratch_transformer_greedy.jsonl"
    },
    [pscustomobject]@{
        Id = "beam4_lp0.8_nr3"
        File = "decoding_beam4_lp0.8_nr3.json"
        Output = "scratch_transformer_beam4_lp0.8_nr3.jsonl"
    },
    [pscustomobject]@{
        Id = "beam4_lp1.0_nr3"
        File = "decoding_beam4_lp1.0_nr3.json"
        Output = "scratch_transformer_beam4_lp1.0_nr3.jsonl"
    }
)

foreach ($item in $configs) {
    $decodingConfig = Join-Path $packageRoot ("configs\" + $item.File)
    $output = Join-Path $outputDir $item.Output

    if (-not (Test-Path -LiteralPath $decodingConfig -PathType Leaf)) {
        throw "Decoding config not found: $decodingConfig"
    }

    Write-Host ""
    Write-Host "=== Running $($item.Id) on $Split ===" -ForegroundColor Cyan

    $inferArgs = @(
        $inferScript,
        "--input", $manifest,
        "--output", $output,
        "--checkpoint", $checkpoint,
        "--tokenizer", $tokenizer,
        "--config", $modelConfig,
        "--decoding-config", $decodingConfig,
        "--batch-size", $BatchSize,
        "--device", $Device,
        "--precision", $Precision,
        "--$Mode"
    )

    & $Python @inferArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Inference failed for $($item.Id) with exit code $LASTEXITCODE."
    }

    Write-Host "=== Validating $($item.Id) ===" -ForegroundColor Yellow
    & $Python $validatorScript `
        --manifest $manifest `
        --predictions $output `
        --expected-config-id $item.Id

    if ($LASTEXITCODE -ne 0) {
        throw "Validation failed for $($item.Id) with exit code $LASTEXITCODE."
    }
}

Write-Host ""
Write-Host "All three configurations completed and passed validation." -ForegroundColor Green
Write-Host "Outputs: $outputDir"
