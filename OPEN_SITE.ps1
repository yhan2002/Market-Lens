[CmdletBinding()]
param([switch]$ValidateOnly)

$ErrorActionPreference = 'Stop'
$configPath = Join-Path $PSScriptRoot 'site_upload_config.json'
$settings = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
$uri = $null

if (
    -not [Uri]::TryCreate([string]$settings.site_url, [UriKind]::Absolute, [ref]$uri) -or
    $uri.Scheme -ne [Uri]::UriSchemeHttps -or
    [string]::IsNullOrWhiteSpace($uri.Host) -or
    -not [string]::IsNullOrWhiteSpace($uri.UserInfo)
) {
    throw 'site_url must be an absolute HTTPS URL without embedded credentials.'
}

if (-not $ValidateOnly) {
    Start-Process -FilePath $uri.AbsoluteUri
}
