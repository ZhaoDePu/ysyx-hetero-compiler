[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$BoardHost,
    [string]$Feed,
    [ValidateSet('rgb', 'jpeg')][string]$FeedFormat = 'rgb',
    [ValidateSet('both', 'en', 'zh')][string]$LabelLanguage = 'both',
    [ValidateRange(1, 65535)][int]$Port = 9000,
    [ValidateRange(1, 65535)][int]$InputPort = 9001,
    [ValidateRange(0, 2147483647)][int]$Frames = 0,
    [switch]$Headless,
    [string]$DetectionsJsonl,
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'
$Viewer = Join-Path $PSScriptRoot 'pc_yolo_viewer_c10.py'
$ViewerArgs = @('--host', $BoardHost, '--port', $Port, '--in-port', $InputPort, '--label-lang', $LabelLanguage)
if ($Feed) { $ViewerArgs += @('--feed', $Feed, '--feed-format', $FeedFormat) }
if ($Frames -gt 0) { $ViewerArgs += @('--frames', $Frames) }
if ($Headless) { $ViewerArgs += '--headless' }
if ($DetectionsJsonl) { $ViewerArgs += @('--detections-jsonl', $DetectionsJsonl) }

& $Python $Viewer @ViewerArgs
exit $LASTEXITCODE
