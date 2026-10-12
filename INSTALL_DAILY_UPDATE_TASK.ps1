param(
    [ValidatePattern("^\d{2}:\d{2}$")]
    [string]$At = "08:30"
)

$ErrorActionPreference = "Stop"
$taskName = "Regime Router Daily Dashboard Update"
$runner = Join-Path $PSScriptRoot "UPDATE_SITE_SCHEDULED.cmd"

if (-not (Test-Path -LiteralPath $runner)) {
    throw "예약 실행 파일을 찾을 수 없습니다: $runner"
}

try {
    $runAt = [datetime]::Today.Add([TimeSpan]::ParseExact($At, "hh\:mm", $null))
} catch {
    throw "-At은 24시간 HH:mm 형식이어야 합니다. 예: 08:30"
}

$days = @(
    [System.DayOfWeek]::Tuesday,
    [System.DayOfWeek]::Wednesday,
    [System.DayOfWeek]::Thursday,
    [System.DayOfWeek]::Friday,
    [System.DayOfWeek]::Saturday
)
$action = New-ScheduledTaskAction `
    -Execute $env:ComSpec `
    -Argument '/d /c UPDATE_SITE_SCHEDULED.cmd' `
    -WorkingDirectory $PSScriptRoot
$trigger = New-ScheduledTaskTrigger -Weekly -WeeksInterval 1 -DaysOfWeek $days -At $runAt
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -WakeToRun `
    -RunOnlyIfNetworkAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 15)
$currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal -UserId $currentUser -LogonType Interactive -RunLevel Limited

try {
    Register-ScheduledTask `
        -TaskName $taskName `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Principal $principal `
        -Description "Cloudium 보유내역 확인, 시장 데이터 갱신, 모델 재계산, 대시보드 게시" `
        -Force | Out-Null
} catch {
    throw "예약 작업 등록에 실패했습니다. 회사 PC 정책 또는 권한을 확인하세요. $($_.Exception.Message)"
}

Write-Host "등록 완료: $taskName"
Write-Host "실행 시각: 화요일~토요일 $At (이 PC의 현지 시각)"
Write-Host "로그: $env:LOCALAPPDATA\RegimeRouter\logs\site_update.log"
