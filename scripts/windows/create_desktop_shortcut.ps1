param(
    [string]$ShortcutName = "NPU Streaming Music Enhancer",
    [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}

$exePath = Join-Path $ProjectRoot "dist\windows\NPUStreamingMusicEnhancer.exe"
if (-not (Test-Path $exePath)) {
    throw "Native EXE was not found: $exePath. Build it first with native\windows\build.ps1 or copy the released EXE into dist\windows."
}

$desktop = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktop "$ShortcutName.lnk"
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $exePath
$shortcut.Arguments = ""
$shortcut.WorkingDirectory = $ProjectRoot
$shortcut.Description = "Native Windows realtime NPU music streaming enhancer for Spotify, Apple Music, and YouTube Music"
$shortcut.IconLocation = "$exePath,0"
$shortcut.Save()

Write-Host "Created Windows desktop shortcut: $shortcutPath"
