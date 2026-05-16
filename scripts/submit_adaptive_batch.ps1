param()

$ErrorActionPreference = "Stop"

function Require-Env($Name) {
    $Value = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($Value)) {
        throw "Set $Name"
    }
    return $Value
}

function Optional-Env($Name, $DefaultValue) {
    $Value = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($Value)) {
        return $DefaultValue
    }
    return $Value
}

function Safe-Name($Value) {
    $Clean = $Value -replace "[^A-Za-z0-9_.-]+", "-"
    $Clean = $Clean.Trim(".-")
    if ($Clean.Length -gt 96) {
        $Clean = $Clean.Substring(0, 96)
    }
    return $Clean
}

$AwsRegion = Require-Env "AWS_REGION"
$BatchJobQueue = Require-Env "BATCH_JOB_QUEUE"
$BatchJobDefinition = Require-Env "BATCH_JOB_DEFINITION"
$S3ConfigsUri = Require-Env "S3_CONFIGS_URI"
$S3DatasetsUri = Require-Env "S3_DATASETS_URI"
$S3OutputsUri = Require-Env "S3_OUTPUTS_URI"

$DatasetConfigs = Optional-Env "DATASET_CONFIGS" "dataset_1_bedroom.json,dataset_2_meeting_room.json"
$AdaptiveMode = Optional-Env "ADAPTIVE_MODE" "metric_conservative"
if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable("EXPERIMENT_COUNT"))) {
    if ($AdaptiveMode -eq "bayes_opt") {
        $ExperimentCount = "24"
    } else {
        $ExperimentCount = "3"
    }
} else {
    $ExperimentCount = [Environment]::GetEnvironmentVariable("EXPERIMENT_COUNT")
}
$BatchRunId = Optional-Env "BATCH_RUN_ID" ([DateTime]::UtcNow.ToString("yyyyMMddTHHmmssZ"))

$TempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("pose-submit-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $TempDir | Out-Null

try {
    foreach ($ConfigName in ($DatasetConfigs -split ",")) {
        $ConfigName = $ConfigName.Trim()
        if ([string]::IsNullOrWhiteSpace($ConfigName)) {
            continue
        }

        $BaseName = [System.IO.Path]::GetFileNameWithoutExtension($ConfigName)
        $SafeDataset = Safe-Name $BaseName
        $OverridePath = Join-Path $TempDir "$SafeDataset.json"

        $Overrides = [ordered]@{
            environment = @(
                [ordered]@{ name = "AWS_REGION"; value = $AwsRegion },
                [ordered]@{ name = "S3_CONFIGS_URI"; value = $S3ConfigsUri },
                [ordered]@{ name = "S3_DATASETS_URI"; value = $S3DatasetsUri },
                [ordered]@{ name = "S3_OUTPUTS_URI"; value = $S3OutputsUri },
                [ordered]@{ name = "CONFIG_NAME"; value = $ConfigName },
                [ordered]@{ name = "EXPERIMENT_COUNT"; value = $ExperimentCount },
                [ordered]@{ name = "ADAPTIVE_MODE"; value = $AdaptiveMode },
                [ordered]@{ name = "BATCH_RUN_ID"; value = $BatchRunId }
            )
        }
        $Overrides | ConvertTo-Json -Depth 10 | Set-Content -Path $OverridePath -Encoding ascii

        $JobName = "pose-$SafeDataset-$BatchRunId"
        $JobId = aws batch submit-job `
            --region $AwsRegion `
            --job-name $JobName `
            --job-queue $BatchJobQueue `
            --job-definition $BatchJobDefinition `
            --container-overrides "file://$OverridePath" `
            --query "jobId" `
            --output text

        if ($LASTEXITCODE -ne 0) {
            throw "Failed to submit $JobName"
        }
        Write-Host "Submitted $JobName`: $JobId"
    }

    Write-Host "Batch run id: $BatchRunId"
} finally {
    Remove-Item -LiteralPath $TempDir -Recurse -Force -ErrorAction SilentlyContinue
}
