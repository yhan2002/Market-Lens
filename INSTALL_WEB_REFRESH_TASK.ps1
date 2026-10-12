$ErrorActionPreference = "Stop"
$taskName = "Regime Router Web Refresh Requests"
$python = Join-Path $PSScriptRoot ".venv\Scripts\pythonw.exe"
$worker = Join-Path $PSScriptRoot "refresh_request_agent.py"
$runner = Join-Path $PSScriptRoot "UPDATE_SITE_SCHEDULED.cmd"
$config = Join-Path $PSScriptRoot "site_upload_config.json"

foreach ($required in @($python, $worker, $runner, $config)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "필수 파일을 찾을 수 없습니다: $required"
    }
}

$action = New-ScheduledTaskAction `
    -Execute $python `
    -Argument ('"{0}"' -f $worker) `
    -WorkingDirectory $PSScriptRoot
$trigger = New-ScheduledTaskTrigger `
    -Once `
    -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes 1)
$trigger.Repetition.StopAtDurationEnd = $false
$settings = New-ScheduledTaskSettingsSet `
    -Hidden `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2)
$currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal -UserId $currentUser -LogonType Interactive -RunLevel Limited

try {
    Register-ScheduledTask `
        -TaskName $taskName `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Principal $principal `
        -Description "웹에서 요청된 최신 데이터 갱신과 대시보드 게시" `
        -Force | Out-Null
} catch {
    throw "예약 작업 등록에 실패했습니다. 회사 PC 정책 또는 권한을 확인하세요. $($_.Exception.Message)"
}

Write-Host "등록 완료: $taskName"
Write-Host "확인 주기: 1분 (이 PC에 현재 사용자가 로그인된 동안)"
Write-Host "실제 갱신 로그: $env:LOCALAPPDATA\RegimeRouter\logs\site_update.log"
