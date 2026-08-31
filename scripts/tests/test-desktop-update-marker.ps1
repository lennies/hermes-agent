# Native Windows regression for the install-global Desktop update lease.

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Handoff = Join-Path $RepoRoot "scripts\desktop-update\windows.ps1"
$Sandbox = Join-Path ([System.IO.Path]::GetTempPath()) ("hermes-marker-test-" + [Guid]::NewGuid().ToString("N"))
$RootInstall = Join-Path $Sandbox "hermes-agent"
$NamedInstall = Join-Path $Sandbox "profiles\delivery-maintainer\hermes-agent"
$Marker = Join-Path $Sandbox ".hermes-update-in-progress"
$PowerShellExe = if ($PSVersionTable.PSEdition -eq "Core") {
    (Get-Command pwsh -ErrorAction Stop).Source
} else {
    Join-Path $PSHOME "powershell.exe"
}
$PythonExe = (Get-Command python -ErrorAction Stop).Source

function Invoke-MarkerSelfTest(
    [string]$InstallRoot,
    [switch]$KeepMarker,
    [switch]$ProbeChildHandoff
) {
    $arguments = @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $Handoff,
        "-InstallRoot", $InstallRoot, "-NoUi", "-SelfTestMarker"
    )
    if ($KeepMarker) { $arguments += "-NoMarkerCleanup" }
    if ($ProbeChildHandoff) {
        $arguments += @("-SelfTestHandoffPython", $PythonExe)
    }
    & $PowerShellExe @arguments | Out-Host
    return $LASTEXITCODE
}

New-Item -ItemType Directory -Path $RootInstall, $NamedInstall -Force | Out-Null
Set-Content -LiteralPath (Join-Path $RootInstall "sentinel.txt") -Value "root" -NoNewline
Set-Content -LiteralPath (Join-Path $NamedInstall "sentinel.txt") -Value "named" -NoNewline

$Holder = $null
try {
    # Hold a real kernel claim open. FileShare.Read permits every product
    # reader, but denies replacement/deletion while this exact handle exists.
    $payloadText = "$PID`n$([DateTimeOffset]::UtcNow.ToUnixTimeSeconds())`nforeign-token`n"
    $payload = [System.Text.Encoding]::ASCII.GetBytes($payloadText)
    $Holder = [System.IO.FileStream]::new(
        $Marker,
        [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::ReadWrite,
        [System.IO.FileShare]::Read
    )
    $Holder.Write($payload, 0, $payload.Length)
    $Holder.Flush($true)

    $rootCode = Invoke-MarkerSelfTest $RootInstall
    if ($rootCode -ne 2) { throw "root contender returned $rootCode, expected 2" }
    $namedCode = Invoke-MarkerSelfTest $NamedInstall
    if ($namedCode -ne 2) { throw "named-profile contender returned $namedCode, expected 2" }

    $Holder.Position = 0
    $observed = New-Object byte[] $payload.Length
    $read = $Holder.Read($observed, 0, $observed.Length)
    if ($read -ne $payload.Length -or $Holder.Length -ne $payload.Length) {
        throw "foreign lease length changed during contention"
    }
    for ($i = 0; $i -lt $payload.Length; $i++) {
        if ($observed[$i] -ne $payload[$i]) {
            throw "foreign lease bytes changed during contention"
        }
    }
    if ((Get-Content -LiteralPath (Join-Path $RootInstall "sentinel.txt") -Raw) -cne "root") {
        throw "root install changed before lease acquisition"
    }
    if ((Get-Content -LiteralPath (Join-Path $NamedInstall "sentinel.txt") -Raw) -cne "named") {
        throw "named install changed before lease acquisition"
    }

    $Holder.Dispose()
    $Holder = $null
    Remove-Item -LiteralPath $Marker -Force

    $productionCode = Invoke-MarkerSelfTest $RootInstall -ProbeChildHandoff
    if ($productionCode -ne 0) { throw "production claim returned $productionCode" }
    if (Test-Path -LiteralPath $Marker) { throw "DeleteOnClose did not release the owned marker" }

    $fixtureCode = Invoke-MarkerSelfTest $NamedInstall -KeepMarker
    if ($fixtureCode -ne 0) { throw "test-fixture claim returned $fixtureCode" }
    $lines = [System.IO.File]::ReadAllLines($Marker)
    if ($lines.Length -ne 3) { throw "published marker did not contain the three-line protocol" }

    Write-Host "DESKTOP UPDATE MARKER TEST: PASS"
} finally {
    if ($Holder) { try { $Holder.Dispose() } catch {} }
    if (Test-Path -LiteralPath $Sandbox) {
        Remove-Item -LiteralPath $Sandbox -Recurse -Force -ErrorAction SilentlyContinue
    }
}
