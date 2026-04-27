param(
    [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$outDir = Join-Path $root "dist\windows"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$source = Join-Path $PSScriptRoot "main.cpp"
$exe = Join-Path $outDir "NPUStreamingMusicEnhancer.exe"

if (-not (Get-Command x86_64-w64-mingw32-g++ -ErrorAction SilentlyContinue)) {
    throw "x86_64-w64-mingw32-g++ was not found. Install mingw-w64 or build with Visual Studio using main.cpp."
}

x86_64-w64-mingw32-g++ -std=c++17 -O2 -municode -mwindows -static $source -o $exe -lgdi32 -lmsimg32 -lcomctl32

Write-Host "Built $exe"
