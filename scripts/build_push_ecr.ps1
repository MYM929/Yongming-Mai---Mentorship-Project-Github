param()

$ErrorActionPreference = "Stop"

function Require-Env($Name) {
    $Value = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($Value)) {
        throw "Set $Name"
    }
    return $Value
}

$AwsRegion = Require-Env "AWS_REGION"
$AwsAccountId = Require-Env "AWS_ACCOUNT_ID"
$EcrRepo = Require-Env "ECR_REPO"
$ImageTag = [Environment]::GetEnvironmentVariable("IMAGE_TAG")
if ([string]::IsNullOrWhiteSpace($ImageTag)) {
    $ImageTag = "pose-arm64-v1"
}

$Registry = "$AwsAccountId.dkr.ecr.$AwsRegion.amazonaws.com"
$ImageUri = "$Registry/$EcrRepo`:$ImageTag"

aws ecr describe-repositories `
    --repository-names $EcrRepo `
    --region $AwsRegion *> $null

if ($LASTEXITCODE -ne 0) {
    aws ecr create-repository `
        --repository-name $EcrRepo `
        --region $AwsRegion *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create ECR repository $EcrRepo"
    }
}

aws ecr get-login-password --region $AwsRegion |
    docker login --username AWS --password-stdin $Registry
if ($LASTEXITCODE -ne 0) {
    throw "Docker login failed"
}

docker buildx build --platform linux/arm64 -t $ImageUri --push .
if ($LASTEXITCODE -ne 0) {
    throw "Docker buildx build failed"
}

Write-Host "Pushed $ImageUri"
