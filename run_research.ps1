# PowerShell equivalent of run_research.sh for Docker Desktop Linux containers.
# Supports Windows PowerShell 5.1 and PowerShell 7.
$ErrorActionPreference = 'Stop'

function Get-EnvironmentOrDefault([string]$Name, [string]$Default) {
    $value = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrEmpty($value)) { return $Default }
    return $value
}

function Get-AbsolutePath([string]$Path) {
    return $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Path)
}

try {
    $projectDir = $PSScriptRoot
    $action = 'help'
    if ($args.Count -gt 0) { $action = [string]$args[0] }
    if ($action -cin @('help', '--help', '-h')) {
        @'
Usage: .\run_research.ps1 [prepare|check|experiment|replay] [OPTIONS]
  prepare     Verify raw recording and annotation identities
  check       Verify fixed controls and synthetic frontend/CNN gradient contracts
  experiment  Run one candidate and print metrics; the agent logs and keeps/discards it
  replay      Inference from --experiment ID --split query|valid|test

Experiment options: --description TEXT [--resume EXPERIMENT_ID]
                    --timeout-seconds N (default 12000; declared candidate budget)
Launcher options: --device auto|cuda|cpu --dry-run
Environment: DATA_DIR, PREPARED_DIR, CACHE_DIR, OUTPUT_DIR, IMAGE_NAME,
             TRAIN_CPU_THREADS (default 4)
Defaults: DATA_DIR=D:\to_Kirst_lab\LFP
          CACHE_DIR=D:\to_Kirst_lab\LFP-cache\meta-learning
          PREPARED_DIR=<CACHE_DIR>\prepared
          OUTPUT_DIR=<script directory>\autoresearch-runs\meta-frontend
          IMAGE_NAME=seizure-prediction-cnn:pytorch
Requires Docker Desktop using Linux containers; CUDA requires GPU support.
All operations run in the foreground. Read program.md before starting research.
Each experiment returns to the agent for result review and the next hypothesis.
The agent repeats until stopped and manages results.tsv and Git commits.
'@ | Write-Output
        exit 0
    }
    if ($action -cnotin @('prepare', 'check', 'experiment', 'replay')) {
        [Console]::Error.WriteLine("Unknown action: $action")
        exit 2
    }

    $device = 'auto'
    $dryRun = $false
    $forwardArgs = @()
    for ($index = 1; $index -lt $args.Count; $index++) {
        $argument = [string]$args[$index]
        if ($argument -ceq '--dry-run') {
            $dryRun = $true
        } elseif ($argument -ceq '--device') {
            if ($index + 1 -ge $args.Count -or
                [string]::IsNullOrEmpty([string]$args[$index + 1]) -or
                ([string]$args[$index + 1]).StartsWith('--')) {
                [Console]::Error.WriteLine('--device requires a value')
                exit 2
            }
            $index++
            $device = [string]$args[$index]
        } else {
            $forwardArgs += $argument
        }
    }

    $dataDir = Get-AbsolutePath (Get-EnvironmentOrDefault 'DATA_DIR' 'D:\to_Kirst_lab\LFP')
    $cacheDir = Get-AbsolutePath (Get-EnvironmentOrDefault 'CACHE_DIR' 'D:\to_Kirst_lab\LFP-cache\meta-learning')
    $preparedDir = Get-AbsolutePath (Get-EnvironmentOrDefault 'PREPARED_DIR' (Join-Path $cacheDir 'prepared'))
    $outputDir = Get-AbsolutePath (Get-EnvironmentOrDefault 'OUTPUT_DIR' (Join-Path $projectDir 'autoresearch-runs\meta-frontend'))
    $imageName = Get-EnvironmentOrDefault 'IMAGE_NAME' 'seizure-prediction-cnn:pytorch'
    $cpuThreads = Get-EnvironmentOrDefault 'TRAIN_CPU_THREADS' '4'
    $preparedMode = 'ro'
    if ($action -ceq 'prepare') { $preparedMode = 'rw' }

    $researchCommit = 'unknown'
    if (Get-Command git -ErrorAction SilentlyContinue) {
        try {
            $commit = & git -C $projectDir rev-parse HEAD 2>$null
            if ($LASTEXITCODE -eq 0 -and $commit) { $researchCommit = [string]$commit }
        } catch {
            # Match the Bash launcher's fallback when Git metadata is unavailable.
            $researchCommit = 'unknown'
        }
    }

    # Docker Desktop manages bind-mount permissions; Windows has no Unix uid/gid.
    $dockerArgs = @('run', '--rm', '--init', '--shm-size=1g',
        '-e', 'HOME=/tmp',
        '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'PYTHONUNBUFFERED=1',
        '-e', "AUTORESEARCH_COMMIT=$researchCommit",
        '-e', "OMP_NUM_THREADS=$cpuThreads", '-e', "MKL_NUM_THREADS=$cpuThreads",
        '-e', "OPENBLAS_NUM_THREADS=$cpuThreads", '-e', 'MPLCONFIGDIR=/tmp/matplotlib',
        '-w', '/workspace', '-v', "${projectDir}:/workspace:ro")
    if ($action -cne 'check') {
        $dockerArgs += @('-v', "${dataDir}:/data/MouseLFP:ro", '-v', "${cacheDir}:/cache",
            '-v', "${preparedDir}:/prepared:${preparedMode}", '-v', "${outputDir}:/output")
    }
    if ($action -cin @('experiment', 'replay')) {
        if ($device -ceq 'cuda' -or $device -clike 'cuda:*' -or
            ($device -ceq 'auto' -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue))) {
            $dockerArgs += @('--gpus', 'all')
        }
    }
    $dockerArgs += $imageName
    switch -CaseSensitive ($action) {
        'prepare' { $dockerArgs += @('python', 'prepare.py', '--data-dir', '/data/MouseLFP', '--output', '/prepared') }
        'check' { $dockerArgs += @('python', 'research.py', 'check') }
        default {
            $dockerArgs += @('python', 'research.py', $action, '--run-dir', '/output',
                '--data-dir', '/data/MouseLFP', '--prepared-dir', '/prepared',
                '--cache-dir', '/cache', '--device', $device)
        }
    }
    $dockerArgs += $forwardArgs

    if ($dryRun) {
        # Single-quoted PowerShell arguments keep spaces and metacharacters literal.
        $quotedArgs = @($dockerArgs | ForEach-Object { "'" + $_.Replace("'", "''") + "'" })
        Write-Output ("& docker " + ($quotedArgs -join ' '))
        exit 0
    }
    if ($action -cne 'check') {
        foreach ($directory in @($cacheDir, $preparedDir, $outputDir)) {
            [System.IO.Directory]::CreateDirectory($directory) | Out-Null
        }
    }
    if ($PSVersionTable.PSVersion -ge [version]'7.3') {
        $PSNativeCommandArgumentPassing = 'Standard'
    } else {
        # Legacy PowerShell reparses native arguments and otherwise loses literal quotes.
        $dockerArgs = @($dockerArgs | ForEach-Object {
            $escaped = [regex]::Replace($_, '(\\*)"', '$1$1\"')
            $escaped = [regex]::Replace($escaped, '(\\+)$', '$1$1')
            '"' + $escaped + '"'
        })
    }
    & docker @dockerArgs
    exit $LASTEXITCODE
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
}
