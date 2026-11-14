param(
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_.@-]*$')]
    [string]$HostAlias = 'msu270',
    [ValidatePattern('^/[A-Za-z0-9_./-]+$')]
    [string]$RuntimeRoot = '/scratch/pr0201/drex_runtime',
    [ValidatePattern('^rollout_audit_[A-Za-z0-9_-]{1,64}$')]
    [string]$RunName = 'rollout_audit_pool64_v1',
    [string]$Destination = 'D:\DriveSuprim_mpailab\results\pool64_v1'
)
$ErrorActionPreference = 'Stop'
$code = @'
import json,sys
from pathlib import Path
run=Path(sys.argv[1])
p=json.loads((run/'latest_result.json').read_text())
result=Path(p['result'])
r=json.loads((result/'report.json').read_text())
if p['remote_exit_code'] != 0 or r['status'] != 'complete':
    raise SystemExit('Audit is not complete yet')
print(result)
'@
$remoteDir = $code | ssh $HostAlias python3 - "$RuntimeRoot/runs/$RunName"
if ($LASTEXITCODE -ne 0) { throw 'Could not locate a completed audit' }
$remoteDir = "$remoteDir".Trim()
$prefix = [regex]::Escape("$RuntimeRoot/runs/$RunName")
if ($remoteDir -notmatch "^$prefix/result_[0-9a-f]{12}$") { throw 'Unexpected result path' }
New-Item -ItemType Directory -Force -Path $Destination | Out-Null
foreach ($name in @('report.json', 'per_scene.csv.gz', 'per_candidate.csv.gz')) {
    scp "${HostAlias}:${remoteDir}/$name" (Join-Path $Destination $name)
    if ($LASTEXITCODE -ne 0) { throw "Download failed: $name" }
}
Write-Host "Downloaded audit to $Destination"
