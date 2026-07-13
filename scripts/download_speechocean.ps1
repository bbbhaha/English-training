param(
    [string]$Destination = "data\raw\speechocean762.tar.gz"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$destinationPath = Join-Path $projectRoot $Destination
$destinationDir = Split-Path -Parent $destinationPath
New-Item -ItemType Directory -Force -Path $destinationDir | Out-Null

$url = "https://www.openslr.org/resources/101/speechocean762.tar.gz"
Write-Host "Downloading SpeechOcean762 (about 520 MB) from OpenSLR..."
& curl.exe --fail --location --retry 3 --output $destinationPath $url
if ($LASTEXITCODE -ne 0) {
    throw "SpeechOcean762 download failed with curl exit code $LASTEXITCODE"
}

$extractDir = Join-Path $projectRoot "data\raw"
Write-Host "Extracting to $extractDir ..."
tar -xzf $destinationPath -C $extractDir
Write-Host "SpeechOcean762 is ready."
