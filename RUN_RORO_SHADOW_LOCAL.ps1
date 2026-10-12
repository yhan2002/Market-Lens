[CmdletBinding()]
param(
    [switch]$Execute,
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]{0,80}$')]
    [string]$RunId = ("roro_shadow_{0}_{1}" -f (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ'), ([guid]::NewGuid().ToString('N').Substring(0, 8))),
    [ValidatePattern('^\d{4}-\d{2}-\d{2}$')]
    [string]$AsOf = '2026-08-12',
    [ValidatePattern('^\d{4}-\d{2}-\d{2}$')]
    [string]$DataStart = '2024-08-01',
    [ValidatePattern('^\d{4}-\d{2}-\d{2}$')]
    [string]$DataEnd = '2026-08-12',
    [string]$FetchAsOf = '2026-08-14T00:00:00Z',
    [ValidateRange(100, 100000)]
    [int]$BootstrapSamples = 500
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function ConvertFrom-IsoDate([string]$Value, [string]$Name) {
    $parsed = [datetime]::MinValue
    if (-not [datetime]::TryParseExact(
        $Value,
        'yyyy-MM-dd',
        [Globalization.CultureInfo]::InvariantCulture,
        [Globalization.DateTimeStyles]::None,
        [ref]$parsed
    )) {
        throw "$Name must be a real yyyy-MM-dd date: $Value"
    }
    return $parsed
}

function Get-HashRecords([string[]]$Paths, [string]$Root) {
    $rootPrefix = [IO.Path]::GetFullPath($Root).TrimEnd('\') + '\'
    foreach ($path in ($Paths | Sort-Object -Unique)) {
        $fullPath = [IO.Path]::GetFullPath($path)
        $displayPath = if ($fullPath.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
            $fullPath.Substring($rootPrefix.Length)
        } else {
            $fullPath
        }
        $stream = [IO.File]::OpenRead($fullPath)
        $sha256 = [Security.Cryptography.SHA256]::Create()
        try {
            $hash = ([BitConverter]::ToString($sha256.ComputeHash($stream))).Replace('-', '')
        } finally {
            $sha256.Dispose()
            $stream.Dispose()
        }
        [pscustomobject][ordered]@{ path = $displayPath; sha256 = $hash }
    }
}

function Get-BundleHash([object[]]$Records) {
    $canonical = (($Records | Sort-Object path | ForEach-Object { "$($_.path)`t$($_.sha256)" }) -join "`n")
    $sha256 = [Security.Cryptography.SHA256]::Create()
    try {
        return ([BitConverter]::ToString(
            $sha256.ComputeHash([Text.Encoding]::UTF8.GetBytes($canonical))
        )).Replace('-', '')
    } finally {
        $sha256.Dispose()
    }
}

$projectRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$outputsRoot = [IO.Path]::GetFullPath((Join-Path $projectRoot 'outputs'))
$stage = [IO.Path]::GetFullPath((Join-Path $outputsRoot $RunId))
$outputsPrefix = $outputsRoot.TrimEnd('\') + '\'
if (-not $stage.StartsWith($outputsPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "staging path escaped outputs: $stage"
}

$asOfDate = ConvertFrom-IsoDate $AsOf 'AsOf'
$dataStartDate = ConvertFrom-IsoDate $DataStart 'DataStart'
$dataEndDate = ConvertFrom-IsoDate $DataEnd 'DataEnd'
if ($dataStartDate -gt $dataEndDate -or $dataEndDate -gt $asOfDate) {
    throw 'require DataStart <= DataEnd <= AsOf'
}
$fetchCutoff = [DateTimeOffset]::Parse(
    $FetchAsOf,
    [Globalization.CultureInfo]::InvariantCulture,
    [Globalization.DateTimeStyles]::RoundtripKind
)
if ($fetchCutoff.Offset -ne [TimeSpan]::Zero -or $fetchCutoff.UtcDateTime.Date -lt $asOfDate.Date) {
    throw 'FetchAsOf must be an aware UTC timestamp on or after AsOf'
}

$publishSource = Join-Path $projectRoot 'regime_router_publish.db'
$privateSource = Join-Path $projectRoot 'regime_router_shadow.db'
foreach ($source in @($publishSource, $privateSource)) {
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "required read-only source DB is missing: $source"
    }
}
$siteRoot = Join-Path $projectRoot 'site'
$requiredSiteInputRelativePaths = @(
    '.openai\hosting.json',
    'build\sites-vite-plugin.ts',
    'cloudflare-env.d.ts',
    'drizzle.config.ts',
    'eslint.config.mjs',
    'next-env.d.ts',
    'next.config.ts',
    'package-lock.json',
    'package.json',
    'postcss.config.mjs',
    'tests\rendered-html.test.mjs',
    'tsconfig.json',
    'vite.config.ts',
    'worker\index.ts'
)
$siteInputDirectories = @('app', 'lib', 'tests', 'build', 'worker', 'db', 'drizzle', 'public')
if ($Execute) {
    if (-not (Test-Path -LiteralPath (Join-Path $siteRoot 'node_modules') -PathType Container)) {
        throw "site dependencies are missing; run npm ci first: $(Join-Path $siteRoot 'node_modules')"
    }
    foreach ($relativePath in $requiredSiteInputRelativePaths) {
        $candidate = Join-Path $siteRoot $relativePath
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            throw "required site build input is missing: $candidate"
        }
    }
    foreach ($directory in $siteInputDirectories) {
        $candidate = Join-Path $siteRoot $directory
        if (-not (Test-Path -LiteralPath $candidate -PathType Container)) {
            throw "required site build input directory is missing: $candidate"
        }
    }
    $requirementsPath = Join-Path $projectRoot 'requirements.txt'
    if (-not (Test-Path -LiteralPath $requirementsPath -PathType Leaf)) {
        throw "Python requirements input is missing: $requirementsPath"
    }
}
if (Test-Path -LiteralPath $stage) {
    throw "fresh staging folder required; already exists: $stage"
}

$plan = [pscustomobject][ordered]@{
    mode = if ($Execute) { 'EXECUTE' } else { 'PLAN_ONLY' }
    run_id = $RunId
    staging_directory = $stage
    as_of = $AsOf
    data_window = "$DataStart..$DataEnd"
    expected_scope = 'HY OAS 330-350 rows; JPY=X/GLD 1,020-1,070 rows; about 150 NQ sessions; two 500-sample bootstraps'
    expected_runtime = '5-12 minutes'
    expected_api_cost_usd = 0
    source_database_policy = 'hash, copy to staging, never write source, verify hash after'
    external_policy = 'network reads only; --no-upload; no deploy'
}
$plan | Format-List | Out-Host
if (-not $Execute) {
    Write-Host "PLAN ONLY: no directory, database, network call, test, upload, or deployment was started."
    Write-Host "Run the same command with -Execute only after explicit approval."
    return
}

$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "project Python is missing: $python"
}
$node = (Get-Command node.exe -ErrorAction Stop).Source
$npm = (Get-Command npm.cmd -ErrorAction Stop).Source

New-Item -ItemType Directory -Path $stage | Out-Null
$logsDirectory = Join-Path $stage 'logs'
New-Item -ItemType Directory -Path $logsDirectory | Out-Null
$runLog = Join-Path $stage 'run.log'
$manifestPath = Join-Path $stage 'run_manifest.json'
$steps = [Collections.Generic.List[object]]::new()

function Write-RunLog([string]$Message) {
    $line = "{0} {1}" -f (Get-Date).ToUniversalTime().ToString('o'), $Message
    Add-Content -LiteralPath $runLog -Value $line -Encoding UTF8
    Write-Host $line
}

function Invoke-LoggedStep(
    [string]$Name,
    [string]$Command,
    [string[]]$Arguments,
    [string]$WorkingDirectory = $projectRoot
) {
    $sequence = $steps.Count + 1
    $safeName = $Name -replace '[^A-Za-z0-9_-]', '_'
    $stepLog = Join-Path $logsDirectory ("{0:D2}_{1}.log" -f $sequence, $safeName)
    $started = (Get-Date).ToUniversalTime()
    $argvJson = @($Command) + @($Arguments) | ConvertTo-Json -Compress
    Write-RunLog "STEP_START name=$Name argv=$argvJson cwd=$WorkingDirectory"
    $exitCode = 1
    $caught = $null
    try {
        Push-Location -LiteralPath $WorkingDirectory
        $global:LASTEXITCODE = 0
        & $Command @Arguments 2>&1 | Tee-Object -FilePath $stepLog | Out-Host
        $exitCode = $LASTEXITCODE
    } catch {
        $caught = $_
        $_ | Out-String | Tee-Object -FilePath $stepLog -Append | Out-Host
    } finally {
        Pop-Location
    }
    $ended = (Get-Date).ToUniversalTime()
    $entry = [pscustomobject][ordered]@{
        sequence = $sequence
        name = $Name
        command = $Command
        argv = @($Arguments)
        working_directory = $WorkingDirectory
        started_utc = $started.ToString('o')
        ended_utc = $ended.ToString('o')
        exit_code = [int]$exitCode
        log_path = $stepLog
    }
    [void]$steps.Add($entry)
    Write-RunLog "STEP_END name=$Name exit_code=$exitCode"
    if ($null -ne $caught) {
        throw $caught
    }
    if ($exitCode -ne 0) {
        throw "step failed with exit code ${exitCode}: $Name"
    }
    return $entry
}

$parameters = [pscustomobject][ordered]@{
    as_of = $AsOf
    display_start = '2026-01-07'
    data_start = $DataStart
    data_end = $DataEnd
    fetch_as_of_utc = $fetchCutoff.ToUniversalTime().ToString('o')
    bootstrap_samples = $BootstrapSamples
    random_seed = 42
    roro_block_days = 40
    sector_block_weeks = 4
    sector_cost_bps = 10
    sector_min_train_weeks = 156
    sector_alpha = 10.0
    sector_top_k = 3
    expected_api_cost_usd = 0
    upload = $false
    deploy = $false
}
$manifest = [ordered]@{
    schema_version = 1
    run_id = $RunId
    status = 'RUNNING'
    started_utc = (Get-Date).ToUniversalTime().ToString('o')
    ended_utc = $null
    project_root = $projectRoot
    staging_directory = $stage
    parameters = $parameters
    environment = $null
    source_hashes_before = @()
    source_hashes_after = @()
    source_unchanged = $null
    clone_hashes = @()
    code_snapshot = $null
    specification_snapshot = $null
    steps = $steps
    output_hashes = @()
    error = $null
    safeguards = @('fresh stage inside outputs', 'source DB copies only', 'stop on nonzero exit', 'no upload', 'no deploy')
}

$deferredError = $null
try {
    $sourceBefore = @(Get-HashRecords @($publishSource, $privateSource) $projectRoot)
    $manifest.source_hashes_before = $sourceBefore

    $rootCode = @(Get-ChildItem -LiteralPath $projectRoot -File | Where-Object {
        $_.Extension -in @('.py', '.ps1')
    } | ForEach-Object FullName)
    $rootCode += Join-Path $projectRoot 'requirements.txt'
    $siteCode = @($requiredSiteInputRelativePaths | ForEach-Object { Join-Path $siteRoot $_ })
    foreach ($directory in $siteInputDirectories) {
        $siteCode += Get-ChildItem -LiteralPath (Join-Path $siteRoot $directory) -Recurse -File |
            Sort-Object FullName |
            ForEach-Object FullName
    }
    $codeHashes = @(Get-HashRecords ($rootCode + $siteCode) $projectRoot)
    $manifest.code_snapshot = [pscustomobject][ordered]@{
        bundle_sha256 = Get-BundleHash $codeHashes
        files = $codeHashes
    }

    $specPaths = @(Get-ChildItem -LiteralPath $projectRoot -File | Where-Object {
        $_.Name -like 'MODEL_SPEC_RORO*.md' -or
        $_.Name -like 'MODEL_SPEC_SECTOR_RORO*.md' -or
        $_.Name -in @('RORO_DATA_CATALOG.md', 'RORO_DATA_CATALOG.csv', 'RORO_SHADOW_RUNBOOK.md')
    } | ForEach-Object FullName)
    $specHashes = @(Get-HashRecords $specPaths $projectRoot)
    $manifest.specification_snapshot = [pscustomobject][ordered]@{
        bundle_sha256 = Get-BundleHash $specHashes
        files = $specHashes
    }

    Write-RunLog "RUN_PARAMETERS $($parameters | ConvertTo-Json -Compress)"
    Write-RunLog "SOURCE_HASHES_BEFORE $($sourceBefore | ConvertTo-Json -Compress)"
    Write-RunLog "CODE_HASHES $($manifest.code_snapshot | ConvertTo-Json -Depth 4 -Compress)"
    Write-RunLog "SPEC_HASHES $($manifest.specification_snapshot | ConvertTo-Json -Depth 4 -Compress)"

    Copy-Item -LiteralPath $publishSource -Destination (Join-Path $stage 'publish_clone.db')
    Copy-Item -LiteralPath $privateSource -Destination (Join-Path $stage 'private_clone.db')
    $manifest.clone_hashes = @(Get-HashRecords @(
        (Join-Path $stage 'publish_clone.db'),
        (Join-Path $stage 'private_clone.db')
    ) $projectRoot)
    Write-RunLog "CLONE_HASHES $($manifest.clone_hashes | ConvertTo-Json -Compress)"

    $pythonVersionStep = Invoke-LoggedStep 'environment_python_version' $python @('--version')
    $nodeVersionStep = Invoke-LoggedStep 'environment_node_version' $node @('--version')
    $npmVersionStep = Invoke-LoggedStep 'environment_npm_version' $npm @('--version')
    $manifest.environment = [pscustomobject][ordered]@{
        os = [Environment]::OSVersion.VersionString
        process_architecture = [Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture.ToString()
        powershell_version = $PSVersionTable.PSVersion.ToString()
        powershell_edition = $PSVersionTable.PSEdition
        timezone = [TimeZoneInfo]::Local.Id
        python = (Get-Content -LiteralPath $pythonVersionStep.log_path -Raw).Trim()
        node = (Get-Content -LiteralPath $nodeVersionStep.log_path -Raw).Trim()
        npm = (Get-Content -LiteralPath $npmVersionStep.log_path -Raw).Trim()
    }
    Write-RunLog "ENVIRONMENT $($manifest.environment | ConvertTo-Json -Compress)"

    $compileFiles = @(Get-ChildItem -LiteralPath $projectRoot -File | Where-Object {
        $_.Name -like 'roro_*.py' -or $_.Name -like 'sector_*shadow*.py'
    } | ForEach-Object FullName)
    Invoke-LoggedStep 'python_compile' $python (@('-m', 'py_compile') + $compileFiles) | Out-Null

    $tests = @(
        'test_roro_data.py', 'test_roro_engine.py', 'test_roro_ew_shadow.py',
        'test_roro_portfolio.py', 'test_roro_axis.py', 'test_roro_phase.py',
        'test_roro_q20.py', 'test_roro_policy.py', 'test_roro_price_observations.py',
        'test_roro_shadow.py', 'test_roro_site_publisher.py', 'test_roro_validation_report.py',
        'test_roro_data_catalog.py',
        'test_validate_roro.py',
        'test_run_roro_shadow_local.py',
        'test_sector_rotation.py', 'test_sector_rotation_forecast.py',
        'test_sector_roro_shadow.py', 'test_sector_shadow_store.py',
        'test_sector_site_publisher.py'
    )
    Invoke-LoggedStep 'focused_python_tests' $python (@('-m', 'unittest') + $tests) | Out-Null
    $catalogPath = Join-Path $projectRoot 'RORO_DATA_CATALOG.csv'
    if (Test-Path -LiteralPath $catalogPath -PathType Leaf) {
        Invoke-LoggedStep 'validate_data_catalog' $python @(
            (Join-Path $projectRoot 'roro_data_catalog.py'), $catalogPath
        ) | Out-Null
    }

    $publishClone = Join-Path $stage 'publish_clone.db'
    $privateClone = Join-Path $stage 'private_clone.db'
    $priceObservations = Join-Path $stage 'roro_price_observations.db'
    $roroShadow = Join-Path $stage 'roro_shadow_r6.db'
    $sectorShadow = Join-Path $stage 'sector_shadow_r6.db'

    Invoke-LoggedStep 'hy_oas_alfred_backfill' $python @(
        (Join-Path $projectRoot 'macro_data.py'), 'backfill-hy-oas-alfred',
        $publishClone, '--start', $DataStart, '--end', $DataEnd
    ) | Out-Null
    Invoke-LoggedStep 'jpy_gold_yahoo_download' $python @(
        (Join-Path $projectRoot 'roro_price_observations.py'), 'download',
        $priceObservations, '--start', $DataStart, '--end', $DataEnd,
        '--as-of', $FetchAsOf
    ) | Out-Null
    Invoke-LoggedStep 'roro_r6_build' $python @(
        (Join-Path $projectRoot 'roro_shadow.py'), 'build',
        $publishClone, $privateClone, $roroShadow,
        '--price-observation-database', $priceObservations, '--as-of', $AsOf
    ) | Out-Null
    Invoke-LoggedStep 'roro_causal_validation' $python @(
        (Join-Path $projectRoot 'roro_validation_report.py'),
        $publishClone, $roroShadow, '--as-of', $AsOf,
        '--bootstrap-samples', [string]$BootstrapSamples,
        '--output-directory', (Join-Path $stage 'validation')
    ) | Out-Null
    Invoke-LoggedStep 'sector_roro_validation' $python @(
        (Join-Path $projectRoot 'sector_roro_shadow.py'), 'validate',
        $publishClone, $roroShadow, '--as-of', $AsOf,
        '--bootstrap-samples', [string]$BootstrapSamples,
        '--output', (Join-Path $stage 'sector_validation.json')
    ) | Out-Null
    Invoke-LoggedStep 'sector_shadow_build' $python @(
        (Join-Path $projectRoot 'sector_shadow_store.py'), 'build',
        $publishClone, $roroShadow, $sectorShadow, '--as-of', $AsOf
    ) | Out-Null
    Invoke-LoggedStep 'roro_payload_no_upload' $python @(
        (Join-Path $projectRoot 'roro_site_publisher.py'),
        '--publish-database', $publishClone, '--private-database', $privateClone,
        '--shadow-database', $roroShadow,
        '--price-observation-database', $priceObservations,
        '--as-of', $AsOf, '--no-upload',
        '--output', (Join-Path $stage 'roro_site_payload.json')
    ) | Out-Null
    Invoke-LoggedStep 'sector_payload_no_upload' $python @(
        (Join-Path $projectRoot 'sector_site_publisher.py'),
        '--shadow-database', $sectorShadow, '--as-of', $AsOf, '--no-upload',
        '--output', (Join-Path $stage 'sector_site_payload.json')
    ) | Out-Null
    Invoke-LoggedStep 'site_tests' $npm @('test') (Join-Path $projectRoot 'site') | Out-Null

    $manifest.status = 'SUCCESS'
} catch {
    $manifest.status = 'FAILED'
    $manifest.error = $_.Exception.Message
    $deferredError = $_
} finally {
    try {
        $sourceAfter = @(Get-HashRecords @($publishSource, $privateSource) $projectRoot)
        $manifest.source_hashes_after = $sourceAfter
        $beforeMap = @{}
        foreach ($item in $manifest.source_hashes_before) { $beforeMap[$item.path] = $item.sha256 }
        $unchanged = $sourceAfter.Count -eq $manifest.source_hashes_before.Count
        foreach ($item in $sourceAfter) {
            if (-not $beforeMap.ContainsKey($item.path) -or $beforeMap[$item.path] -ne $item.sha256) {
                $unchanged = $false
            }
        }
        $manifest.source_unchanged = $unchanged
        if (-not $unchanged) {
            $manifest.status = 'FAILED'
            $manifest.error = 'operating source DB hash changed during SHADOW run'
            if ($null -eq $deferredError) { $deferredError = [Exception]$manifest.error }
        }
        Write-RunLog "SOURCE_HASHES_AFTER $($sourceAfter | ConvertTo-Json -Compress)"
    } catch {
        $manifest.status = 'FAILED'
        $manifest.source_unchanged = $false
        $manifest.error = "source hash verification failed: $($_.Exception.Message)"
        if ($null -eq $deferredError) { $deferredError = $_ }
    }

    $outputFiles = @(Get-ChildItem -LiteralPath $stage -Recurse -File | Where-Object {
        $_.Name -ne 'run_manifest.json' -and $_.Extension -ne '.log'
    } | ForEach-Object FullName)
    $manifest.output_hashes = @(Get-HashRecords $outputFiles $projectRoot)
    $manifest.steps = @($steps)
    $manifest.ended_utc = (Get-Date).ToUniversalTime().ToString('o')
    Write-RunLog "OUTPUT_HASHES $($manifest.output_hashes | ConvertTo-Json -Compress)"
    Write-RunLog "RUN_END status=$($manifest.status) source_unchanged=$($manifest.source_unchanged)"
    $manifest | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
}

if ($null -ne $deferredError) {
    throw $deferredError
}
Write-Host "SHADOW run complete: $stage"
Write-Host "Manifest: $manifestPath"
