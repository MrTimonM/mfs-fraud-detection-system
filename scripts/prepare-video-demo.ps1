param([string]$BaseUrl = 'http://localhost:3000')

$ErrorActionPreference = 'Stop'
$demoUri = [uri]$BaseUrl
if ($demoUri.Host -notin @('localhost', '127.0.0.1', '::1')) {
  throw 'Prepare the recording on a local demo server. The public Vercel demo can reset between requests.'
}
$demoWorkspace = Invoke-RestMethod "$BaseUrl/api/v1/workspace"
if ($demoWorkspace.storage_mode -eq 'postgres') {
  throw 'Use a demo workspace to prepare synthetic recording data.'
}
$demoAccount = 'video-wallet-' + [guid]::NewGuid().ToString('N').Substring(0, 8)
$demoNow = [DateTimeOffset]::UtcNow
for ($demoIndex = 0; $demoIndex -lt 48; $demoIndex++) {
  $demoPayload = @{
    transaction_id = "$demoAccount-history-$demoIndex"
    user_id = $demoAccount
    transaction_type = 'SEND_MONEY'
    amount = 800 + (($demoIndex * 137) % 800)
    balance_before = 15000
    timestamp = $demoNow.AddHours(-12 * (48 - $demoIndex)).UtcDateTime.ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
    device_id = "$demoAccount-device"
    receiver_id = "$demoAccount-recipient"
    channel = 'APP'
  }
  $null = Invoke-RestMethod "$BaseUrl/api/v1/transactions/analyze" -Method Post -ContentType 'application/json' -Body ($demoPayload | ConvertTo-Json)
}
$demoCandidate = @{
  transaction_id = "$demoAccount-unusual-transfer"
  user_id = $demoAccount
  transaction_type = 'SEND_MONEY'
  amount = 9000
  balance_before = 15000
  timestamp = $demoNow.UtcDateTime.ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
  device_id = "$demoAccount-device"
  receiver_id = "$demoAccount-recipient"
  channel = 'APP'
}
$demoResult = Invoke-RestMethod "$BaseUrl/api/v1/transactions/analyze" -Method Post -ContentType 'application/json' -Body ($demoCandidate | ConvertTo-Json)
if ($demoResult.decision -ne 'APPROVE' -or -not $demoResult.anomaly.review_recommended) {
  throw "The rehearsed contrast is unavailable with this history/policy. Inspect the returned evidence and existing custom rules before recording. Rule decision: $($demoResult.decision); anomaly score: $($demoResult.anomaly.score)."
}
Write-Host 'Prepared real application evidence from synthetic account history.'
Write-Host "Account: $demoAccount"
Write-Host "Rule decision: $($demoResult.decision); rule score: $($demoResult.risk_score)"
Write-Host "Anomaly score: $($demoResult.anomaly.score); threshold: $($demoResult.anomaly.threshold)"
Write-Host "Open for recording: $BaseUrl/cases/case-$($demoResult.id)"
Write-Host 'Keep this server running. The history is synthetic; this is not evidence of real customer fraud.'
