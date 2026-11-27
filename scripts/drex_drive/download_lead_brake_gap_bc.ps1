[CmdletBinding()]
param(
    [Parameter(Mandatory)][ValidatePattern('^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$')][string]$MsuAlias,
    [Parameter(Mandatory)][ValidatePattern('^(/[A-Za-z0-9_][A-Za-z0-9_.-]*)+$')][string]$MsuRuntimeRoot,
    [Parameter(Mandatory)][ValidatePattern('^attempt-[0-9]{3,}$')][string]$Attempt,
    [Parameter(Mandatory)][string]$LocalParent,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$experiment = 'corridor-preview-lead-brake-gap-conditional-v1'
$root = "$MsuRuntimeRoot/runs/$experiment/$Attempt"
$parent = Get-Item -LiteralPath $LocalParent
if (($parent -isnot [IO.DirectoryInfo]) -or ($parent.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
    throw 'LocalParent must be an existing normal directory.'
}
$destination = Join-Path $parent.FullName "$experiment-$Attempt"
if (Test-Path -LiteralPath $destination) { throw "Destination already exists: $destination" }
$fixed = @('manifest.json', 'smoke-score.json', 'comparison.json', 'bc-score-smoke.log', 'bc-score.log',
    'bc-smoke/runtime.json', 'bc-smoke/prediction-receipt.json',
    'bc-pilot/runtime.json', 'bc-pilot/prediction-receipt.json',
    'brake-gap-bc-smoke-slurm/submit.log', 'brake-gap-bc-predict-slurm/submit.log')
$logPattern = '^brake-gap-bc-(smoke|predict)-slurm/slurm-[0-9]+(\.stdout|\.stderr)?\.log$'
if ($DryRun) {
    Write-Output "No connection or writes: ${MsuAlias}:$root -> $destination"
    Write-Output $fixed
    Write-Output $logPattern
    return
}

# Query only the explicit attempt and the named file patterns; do not recurse.
# Encode the constant inventory program to avoid native Windows argument quoting.
$list = @'
set -euo pipefail
root=$1
[[ -d "$root" && "$root" == "$(realpath -e -- "$root")" ]]
cd -- "$root"
shopt -s nullglob
for p in manifest.json smoke-score.json comparison.json bc-score-smoke.log bc-score.log \
         bc-{smoke,pilot}/{runtime,prediction-receipt}.json \
         brake-gap-bc-{smoke,predict}-slurm/{submit.log,slurm-*.log}; do
  case "$p" in
    */slurm-*) [[ "$p" =~ ^brake-gap-bc-(smoke|predict)-slurm/slurm-[0-9]+(\.stdout|\.stderr)?\.log$ ]] || continue ;;
  esac
  if [[ -f "$p" && "$root/$p" == "$(realpath -e -- "$p")" ]]; then
    printf '%s\n' "$p"
  fi
done
'@
$program = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes(($list -replace "`r", '')))
$inventory = @(& ssh -T -o BatchMode=yes -- $MsuAlias "printf %s $program | base64 -d | bash -s -- '$root'")
if ($LASTEXITCODE -ne 0) { throw "MSU inventory failed (ssh exit $LASTEXITCODE); no files copied." }
foreach ($item in $inventory) {
    if (($fixed -cnotcontains $item) -and ($item -cnotmatch $logPattern)) {
        throw "Unexpected inventory entry: $item"
    }
}
if ($inventory.Count -eq 0) { throw 'No allowlisted files available; no directory created.' }
if (@($inventory | Sort-Object -Unique).Count -ne $inventory.Count) { throw 'Duplicate inventory entries.' }
New-Item -ItemType Directory -Path $destination | Out-Null
$failed = @()
foreach ($item in ($inventory | Sort-Object)) {
    $target = Join-Path $destination ($item.Replace('/', [IO.Path]::DirectorySeparatorChar))
    $targetParent = Split-Path -Parent $target
    if (-not (Test-Path -LiteralPath $targetParent)) {
        New-Item -ItemType Directory -Path $targetParent | Out-Null
    }
    if (Test-Path -LiteralPath $target) { throw "Refusing overwrite: $target" }
    & scp -o BatchMode=yes -- "${MsuAlias}:$root/$item" $target
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Copy failed for $item (scp exit $LASTEXITCODE); partial copy kept."
        $failed += $item
    }
}
foreach ($missing in ($fixed | Where-Object { $inventory -cnotcontains $_ })) {
    Write-Warning "Not present in this attempt: $missing"
}
if ($failed.Count) { throw "Partial download: $($inventory.Count - $failed.Count)/$($inventory.Count); see $destination" }
Write-Output "Downloaded $($inventory.Count) allowlisted files to $destination"
