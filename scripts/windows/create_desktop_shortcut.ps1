param(
    [string]$ShortcutName = "NPU Streaming Music Enhancer",
    [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}

$pythonw = (Get-Command pythonw.exe -ErrorAction SilentlyContinue)
if ($null -eq $pythonw) {
    $python = Get-Command python.exe -ErrorAction Stop
    $pythonwPath = $python.Source
} else {
    $pythonwPath = $pythonw.Source
}

Push-Location $ProjectRoot
try {
    & python -m pip install -e .
} finally {
    Pop-Location
}

$desktop = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktop "$ShortcutName.lnk"
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $pythonwPath
$shortcut.Arguments = "-m npu_audio_enhancer.desktop"
$shortcut.WorkingDirectory = $ProjectRoot
$shortcut.Description = "Realtime NPU music streaming enhancer for Spotify, Apple Music, and YouTube Music"
$shortcut.IconLocation = "$pythonwPath,0"
$shortcut.Save()

Write-Host "Created Windows desktop shortcut: $shortcutPath"
