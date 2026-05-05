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

function Assert-Real-Value($Name, $Value) {
    if ($Value -match "<.*>" -or $Value -match "\bxxx\b" -or $Value -match "\baaa\b" -or $Value -match "\byyy\b") {
        throw "$Name is still a placeholder: $Value"
    }
}

function Split-Csv($Value) {
    return @($Value -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })
}

function Get-Name-From-Arn($Arn) {
    return ($Arn -split "/")[-1]
}

function Invoke-Aws($Description, [scriptblock]$Command) {
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw $Description
    }
}

function Wait-ComputeEnvironmentValid($Name, $Region) {
    $Deadline = (Get-Date).AddMinutes(10)
    while ((Get-Date) -lt $Deadline) {
        $Status = aws batch describe-compute-environments `
            --region $Region `
            --compute-environments $Name `
            --query "computeEnvironments[0].status" `
            --output text
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to describe compute environment $Name"
        }

        $Reason = aws batch describe-compute-environments `
            --region $Region `
            --compute-environments $Name `
            --query "computeEnvironments[0].statusReason" `
            --output text
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to describe compute environment $Name"
        }

        Write-Host "Compute environment $Name status: $Status ($Reason)"
        if ($Status -eq "VALID") {
            return
        }
        if ($Status -eq "INVALID") {
            throw "Compute environment $Name is INVALID: $Reason"
        }
        Start-Sleep -Seconds 15
    }

    throw "Timed out waiting for compute environment $Name to become VALID"
}

$AwsRegion = Require-Env "AWS_REGION"
$AwsAccountId = Require-Env "AWS_ACCOUNT_ID"
$EcrRepo = Require-Env "ECR_REPO"
$BatchInstanceRoleArn = Require-Env "BATCH_INSTANCE_ROLE_ARN"
$BatchJobRoleArn = Require-Env "BATCH_JOB_ROLE_ARN"
$SubnetIds = Require-Env "SUBNET_IDS"
$SecurityGroupIds = Require-Env "SECURITY_GROUP_IDS"

Assert-Real-Value "BATCH_INSTANCE_ROLE_ARN" $BatchInstanceRoleArn
Assert-Real-Value "BATCH_JOB_ROLE_ARN" $BatchJobRoleArn
Assert-Real-Value "SUBNET_IDS" $SubnetIds
Assert-Real-Value "SECURITY_GROUP_IDS" $SecurityGroupIds

$BatchInstanceProfileName = Get-Name-From-Arn $BatchInstanceRoleArn
$BatchJobRoleName = Get-Name-From-Arn $BatchJobRoleArn

Invoke-Aws "Instance profile not found: $BatchInstanceProfileName" {
    aws iam get-instance-profile `
        --instance-profile-name $BatchInstanceProfileName *> $null
}

Invoke-Aws "Batch job role not found: $BatchJobRoleName" {
    aws iam get-role `
        --role-name $BatchJobRoleName *> $null
}

$SubnetIdList = Split-Csv $SubnetIds
$SecurityGroupIdList = Split-Csv $SecurityGroupIds

Invoke-Aws "One or more SUBNET_IDS are invalid: $SubnetIds" {
    aws ec2 describe-subnets `
        --region $AwsRegion `
        --subnet-ids $SubnetIdList *> $null
}

Invoke-Aws "One or more SECURITY_GROUP_IDS are invalid: $SecurityGroupIds" {
    aws ec2 describe-security-groups `
        --region $AwsRegion `
        --group-ids $SecurityGroupIdList *> $null
}

$BatchServiceRoleArn = [Environment]::GetEnvironmentVariable("BATCH_SERVICE_ROLE_ARN")
if ([string]::IsNullOrWhiteSpace($BatchServiceRoleArn)) {
    $BatchServiceRoleArn = aws iam get-role `
        --role-name AWSServiceRoleForBatch `
        --query "Role.Arn" `
        --output text 2>$null

    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($BatchServiceRoleArn) -or $BatchServiceRoleArn -eq "None") {
        aws iam create-service-linked-role --aws-service-name batch.amazonaws.com *> $null
        if ($LASTEXITCODE -ne 0) {
            throw "Set BATCH_SERVICE_ROLE_ARN or allow iam:CreateServiceLinkedRole for batch.amazonaws.com"
        }
        $BatchServiceRoleArn = aws iam get-role `
            --role-name AWSServiceRoleForBatch `
            --query "Role.Arn" `
            --output text
        if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($BatchServiceRoleArn) -or $BatchServiceRoleArn -eq "None") {
            throw "Could not resolve AWSServiceRoleForBatch ARN"
        }
    }
    Write-Host "Using Batch service role $BatchServiceRoleArn"
}

$ImageTag = Optional-Env "IMAGE_TAG" "pose-arm64-v1"
$ImageUri = "$AwsAccountId.dkr.ecr.$AwsRegion.amazonaws.com/$EcrRepo`:$ImageTag"

$BatchComputeEnv = Optional-Env "BATCH_COMPUTE_ENV" "pose-tracking-arm64-ce"
$BatchJobQueue = Optional-Env "BATCH_JOB_QUEUE" "pose-tracking-arm64-queue"
$BatchJobDefinition = Optional-Env "BATCH_JOB_DEFINITION" "pose-tracking-arm64"
$BatchMaxVcpus = [int](Optional-Env "BATCH_MAX_VCPUS" "8")
$BatchJobVcpus = Optional-Env "BATCH_JOB_VCPUS" "4"
$BatchJobMemoryMb = Optional-Env "BATCH_JOB_MEMORY_MB" "7000"
$BatchInstanceTypes = Optional-Env "BATCH_INSTANCE_TYPES" "c7g.xlarge,c6g.xlarge"

$TempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("pose-batch-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $TempDir | Out-Null

try {
    $ComputeResourcesPath = Join-Path $TempDir "compute_resources.json"
    $ContainerPropertiesPath = Join-Path $TempDir "container_properties.json"

    $ComputeResources = [ordered]@{
        type = "EC2"
        allocationStrategy = "BEST_FIT_PROGRESSIVE"
        minvCpus = 0
        maxvCpus = $BatchMaxVcpus
        desiredvCpus = 0
        instanceTypes = @(Split-Csv $BatchInstanceTypes)
        subnets = @(Split-Csv $SubnetIds)
        securityGroupIds = @(Split-Csv $SecurityGroupIds)
        instanceRole = $BatchInstanceRoleArn
    }
    $ComputeResources | ConvertTo-Json -Depth 10 | Set-Content -Path $ComputeResourcesPath -Encoding ascii

    $ContainerProperties = [ordered]@{
        image = $ImageUri
        jobRoleArn = $BatchJobRoleArn
        resourceRequirements = @(
            [ordered]@{ type = "VCPU"; value = $BatchJobVcpus },
            [ordered]@{ type = "MEMORY"; value = $BatchJobMemoryMb }
        )
    }
    $ContainerProperties | ConvertTo-Json -Depth 10 | Set-Content -Path $ContainerPropertiesPath -Encoding ascii

    $ComputeEnvExists = aws batch describe-compute-environments `
        --region $AwsRegion `
        --compute-environments $BatchComputeEnv `
        --query "length(computeEnvironments)" `
        --output text

    if ($ComputeEnvExists -eq "0") {
        aws batch create-compute-environment `
            --region $AwsRegion `
            --compute-environment-name $BatchComputeEnv `
            --type MANAGED `
            --state ENABLED `
            --service-role $BatchServiceRoleArn `
            --compute-resources "file://$ComputeResourcesPath"
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create compute environment $BatchComputeEnv"
        }
        Write-Host "Created compute environment $BatchComputeEnv"
    } else {
        aws batch update-compute-environment `
            --region $AwsRegion `
            --compute-environment $BatchComputeEnv `
            --state ENABLED `
            --compute-resources "maxvCpus=$BatchMaxVcpus"
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to update compute environment $BatchComputeEnv"
        }
        Write-Host "Updated compute environment $BatchComputeEnv"
    }

    Wait-ComputeEnvironmentValid $BatchComputeEnv $AwsRegion

    $QueueExists = aws batch describe-job-queues `
        --region $AwsRegion `
        --job-queues $BatchJobQueue `
        --query "length(jobQueues)" `
        --output text

    if ($QueueExists -eq "0") {
        aws batch create-job-queue `
            --region $AwsRegion `
            --job-queue-name $BatchJobQueue `
            --state ENABLED `
            --priority 1 `
            --compute-environment-order "order=1,computeEnvironment=$BatchComputeEnv"
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create job queue $BatchJobQueue"
        }
        Write-Host "Created job queue $BatchJobQueue"
    } else {
        aws batch update-job-queue `
            --region $AwsRegion `
            --job-queue $BatchJobQueue `
            --state ENABLED `
            --priority 1 `
            --compute-environment-order "order=1,computeEnvironment=$BatchComputeEnv"
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to update job queue $BatchJobQueue"
        }
        Write-Host "Updated job queue $BatchJobQueue"
    }

    $Revision = aws batch register-job-definition `
        --region $AwsRegion `
        --job-definition-name $BatchJobDefinition `
        --type container `
        --platform-capabilities EC2 `
        --container-properties "file://$ContainerPropertiesPath" `
        --query "revision" `
        --output text
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to register job definition $BatchJobDefinition"
    }

    Write-Host "Registered job definition $BatchJobDefinition`:$Revision"
    Write-Host "Image: $ImageUri"
    Write-Host "Queue: $BatchJobQueue"
} finally {
    Remove-Item -LiteralPath $TempDir -Recurse -Force -ErrorAction SilentlyContinue
}
