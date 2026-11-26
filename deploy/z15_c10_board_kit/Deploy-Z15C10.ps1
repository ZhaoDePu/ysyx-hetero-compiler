[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^(?:[A-Za-z0-9][A-Za-z0-9._-]*@)?[A-Za-z0-9][A-Za-z0-9._-]*$')]
    [string]$BoardHost,
    [ValidateRange(1, 65535)][int]$SshPort = 22,
    [ValidateSet('Bench', 'Tcp', 'V4l2')][string]$Action = 'Bench',
    [ValidateSet('auto', '125_100', '200_150')][string]$FreqProfile = 'auto',
    [ValidateRange(1, 1000)][int]$SteadyFrames = 10,
    [ValidatePattern('^/dev/video[0-9]{1,4}$')][string]$Camera = '/dev/video0',
    [ValidateRange(1, 1920)][int]$Width = 640,
    [ValidateRange(1, 1080)][int]$Height = 480,
    [ValidateRange(1, 120)][int]$Fps = 30,
    [string]$Feed,
    [ValidateSet('rgb', 'jpeg')][string]$FeedFormat = 'rgb',
    [ValidateSet('both', 'en', 'zh')][string]$LabelLanguage = 'both',
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]*$')][string]$ViewerHost,
    [ValidateRange(1, 65535)][int]$Port = 9000,
    [ValidateRange(1, 65535)][int]$InputPort = 9001,
    [ValidateRange(0, 2147483647)][int]$LiveFrames = 0,
    [string]$Python = 'python',
    [string]$Archive = (Join-Path $PSScriptRoot 'z15_c10_board_kit.tar.gz'),
    [string]$OutputRoot = (Join-Path $PSScriptRoot 'results'),
    [switch]$Plan
)

$ErrorActionPreference = 'Stop'
if ($Action -eq 'Tcp' -and [string]::IsNullOrWhiteSpace($Feed)) {
    throw 'Tcp requires -Feed with a Windows camera index, image, or video path.'
}
if ($Action -ne 'Tcp' -and $Feed) {
    throw '-Feed is valid only with -Action Tcp.'
}
if ($Action -eq 'Tcp' -and $Port -eq $InputPort) {
    throw 'TCP input and output ports must be different.'
}
if (-not $ViewerHost) {
    $ViewerHost = ($BoardHost -split '@')[-1]
}

$RunId = 'z15-c10-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' +
    [Guid]::NewGuid().ToString('N').Substring(0, 12)
$RemoteRoot = '/tmp/' + $RunId
$RemoteKit = $RemoteRoot + '/z15_c10_board_kit'
$RemoteReports = $RemoteRoot + '/reports'
$SummaryNames = @('board_c10_summary.csv', 'sim_vs_board_comparison.csv')
if ($Plan) {
    [pscustomobject]@{
        Host = $BoardHost
        Action = $Action
        FreqProfile = $FreqProfile
        Stage = $RemoteRoot
        Sequence = 'FPGA Manager check, upload and SHA-256, MMIO MAGIC probe, benchmark, pull CSVs'
        Viewer = if ($Action -ne 'Bench') { $ViewerHost } else { $null }
    }
    return
}

$Ssh = (Get-Command ssh.exe -CommandType Application).Source
$Scp = (Get-Command scp.exe -CommandType Application).Source
$SshOptions = @('-p', $SshPort, '-o', 'ConnectTimeout=10', '-o', 'ClearAllForwardings=yes')
$ScpOptions = @('-O', '-P', $SshPort, '-o', 'ConnectTimeout=10', '-o', 'ClearAllForwardings=yes')
$Archive = (Resolve-Path -LiteralPath $Archive).ProviderPath
$Manifest = Join-Path ([System.IO.Path]::GetDirectoryName($Archive)) 'SHA256SUMS.txt'
$ArchiveName = [System.IO.Path]::GetFileName($Archive)
$NamePattern = [regex]::Escape($ArchiveName)
$ManifestRows = @(Get-Content -LiteralPath $Manifest | Where-Object {
    $_ -match ('^[0-9a-fA-F]{64} [ *](?:\./)?' + $NamePattern + '$')
})
if ($ManifestRows.Count -ne 1) {
    throw "Expected one SHA-256 entry for $ArchiveName in $Manifest."
}
$ExpectedHash = $ManifestRows[0].Substring(0, 64).ToLowerInvariant()
if ((Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ExpectedHash) {
    throw "Archive SHA-256 mismatch: $ArchiveName"
}
if ($Action -ne 'Bench') {
    $Viewer = Join-Path $PSScriptRoot 'pc_yolo_viewer_c10.py'
    if (-not (Test-Path -LiteralPath $Viewer -PathType Leaf)) {
        throw "Missing viewer: $Viewer"
    }
    $null = Get-Command $Python -ErrorAction Stop
}
$LocalReports = Join-Path $OutputRoot $RunId
$null = New-Item -ItemType Directory -Path $LocalReports

function Invoke-Board([string]$Command) {
    $Output = & $Ssh @SshOptions $BoardHost $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Board command failed with exit code $LASTEXITCODE. Run directory: $RemoteRoot"
    }
    return $Output
}

function Receive-BoardFile([string]$RemoteDirectory, [string]$Name) {
    $RemoteFile = '{0}:{1}/{2}' -f $BoardHost, $RemoteDirectory, $Name
    $LocalFile = Join-Path $LocalReports $Name
    & $Scp @ScpOptions $RemoteFile $LocalFile
    if ($LASTEXITCODE -ne 0) {
        throw "Could not retrieve $Name from $RemoteDirectory"
    }
}

$ManagerCheck = 'test "$(cat /sys/class/fpga_manager/fpga0/state)" = operating'
Invoke-Board "set -eu; $ManagerCheck; umask 077; mkdir $RemoteRoot"
& $Scp @ScpOptions $Archive ('{0}:{1}/kit.tar.gz' -f $BoardHost, $RemoteRoot)
if ($LASTEXITCODE -ne 0) {
    throw "Archive upload failed. Run directory: $RemoteRoot"
}
$VerifyKit = "cd $RemoteRoot; echo '$ExpectedHash  kit.tar.gz' | sha256sum -c -; " +
    "tar -xzf kit.tar.gz; cd $RemoteKit; sha256sum -c SHA256SUMS.txt"
Invoke-Board "set -eu; $ManagerCheck; $VerifyKit"
$Probe = "$RemoteKit/run_z15_c10_bench.sh --probe-only"
Invoke-Board "set -eu; $Probe"
$ProfileOutput = @(Invoke-Board "set -eu; $RemoteKit/run_z15_c10_bench.sh --profile-only --freq-profile $FreqProfile")
if ($ProfileOutput.Count -ne 1 -or $ProfileOutput[0].Trim() -notin @('125_100', '200_150')) {
    throw 'The board did not report a supported clock pair.'
}
$MatchedProfile = $ProfileOutput[0].Trim()
$Bench = "$RemoteKit/run_z15_c10_bench.sh --freq-profile $MatchedProfile " +
    "--steady-frames $SteadyFrames --out $RemoteReports"
Invoke-Board "set -eu; $Bench"
foreach ($Name in $SummaryNames) {
    Receive-BoardFile $RemoteReports $Name
}
Write-Output "Benchmark CSVs: $LocalReports"
if ($Action -eq 'Bench') {
    return
}

$Source = $Action.ToLowerInvariant()
$Serve = 'set -eu; ' + $ManagerCheck + "; cd $RemoteKit; exec ./bin/z15_yolo_live_c10 " +
    "--model models/yolo26n/program.yhb --source $Source --live " +
    "--freq-profile $MatchedProfile --csv $RemoteRoot/live.csv --port $Port --in-port $InputPort"
if ($LiveFrames -gt 0) {
    $Serve += " --frames $LiveFrames"
}
if ($Action -eq 'V4l2') {
    $Serve += " --device $Camera --width $Width --height $Height --fps $Fps"
}
# A PTY ties the foreground board service to this SSH connection.
$SshArgs = @('-tt', '-o', 'BatchMode=yes') + $SshOptions + @($BoardHost, ('"' + $Serve.Replace('"', '\"') + '"'))
$StartOptions = @{
    FilePath = $Ssh
    ArgumentList = $SshArgs
    WindowStyle = 'Hidden'
    PassThru = $true
    RedirectStandardOutput = (Join-Path $LocalReports 'service.stdout.log')
    RedirectStandardError = (Join-Path $LocalReports 'service.stderr.log')
}
$Remote = Start-Process @StartOptions
try {
    $ViewerArgs = @('--host', $ViewerHost, '--port', $Port, '--in-port', $InputPort,
                    '--label-lang', $LabelLanguage)
    if ($Action -eq 'Tcp') {
        $ViewerArgs += @('--feed', $Feed, '--feed-format', $FeedFormat)
    }
    if ($LiveFrames -gt 0) {
        $ViewerArgs += @('--frames', $LiveFrames)
    }
    & $Python $Viewer @ViewerArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Viewer failed. Service logs: $LocalReports"
    }
}
finally {
    if (-not $Remote.HasExited) {
        Stop-Process -Id $Remote.Id -ErrorAction SilentlyContinue
        $null = $Remote.WaitForExit(3000)
    }
    $Remote.Dispose()
}
Write-Output "Board live log: $RemoteRoot/live.csv"
