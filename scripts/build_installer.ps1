<#
.SYNOPSIS
    Builds the Qwen-Image-Edit Windows installer using Inno Setup.
.DESCRIPTION
    1. Downloads uv.exe (Windows x86_64) if not present
    2. Creates icon.ico from a built-in PNG if icon.ico doesn't exist
    3. Runs Inno Setup Compiler (iscc) to build the installer
    4. Output is placed in scripts\output\
.PARAMETER SkipUvDownload
    Skip downloading uv.exe if already present
.PARAMETER LzmaThreads
    Number of threads for LZMA compression (default: auto)
.EXAMPLE
    .\build_installer.ps1
    .\build_installer.ps1 -LzmaThreads 4
#>

param(
    [switch]$SkipUvDownload,
    [int]$LzmaThreads = 0
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $PSCommandPath
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
$RedistDir = Join-Path $ScriptDir "redist"
$OutputDir = Join-Path $ScriptDir "output"

# Ensure directories exist
New-Item -ItemType Directory -Path $RedistDir -Force | Out-Null
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

Write-Host "╔══════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║      Qwen-Image-Edit - Windows Installer Builder ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

<# Step 1: Download uv.exe #>
Write-Host "[1/4] Verificando uv.exe..." -ForegroundColor Yellow
$UvPath = Join-Path $RedistDir "uv.exe"
if (-not (Test-Path $UvPath)) {
    Write-Host "      Descargando uv.exe desde astral.sh..." -ForegroundColor Yellow
    $UvUrl = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
    $TempZip = Join-Path $env:TEMP "uv.zip"
    try {
        Invoke-WebRequest -Uri $UvUrl -OutFile $TempZip -UseBasicParsing
        Expand-Archive -Path $TempZip -DestinationPath $RedistDir -Force
        if (-not (Test-Path $UvPath)) {
            # uv might be in a subfolder
            $extracted = Get-ChildItem $RedistDir -Recurse -Filter "uv.exe" | Select-Object -First 1
            if ($extracted) {
                Copy-Item $extracted.FullName $UvPath -Force
            }
        }
        Remove-Item $TempZip -Force -ErrorAction SilentlyContinue
        Write-Host "      [OK] uv.exe descargado" -ForegroundColor Green
    }
    catch {
        Write-Host "      [ERROR] No se pudo descargar uv.exe: $_" -ForegroundColor Red
        Write-Host "      Descarguelo manualmente de $UvUrl" -ForegroundColor Yellow
        Write-Host "      y coloquelo en: $UvPath" -ForegroundColor Yellow
        exit 1
    }
}
else {
    $uvVer = & $UvPath --version 2>$null
    Write-Host "      [OK] uv.exe encontrado: $uvVer" -ForegroundColor Green
}

<# Step 2: Check icon.ico #>
Write-Host "[2/4] Verificando icono..." -ForegroundColor Yellow
$IconPath = Join-Path $ScriptDir "icon.ico"
if (-not (Test-Path $IconPath)) {
    Write-Host "      [INFO] icon.ico no encontrado. Se usara el icono por defecto de Inno Setup." -ForegroundColor Yellow
    Write-Host "      Para personalizarlo, coloque un archivo icon.ico en:" -ForegroundColor Yellow
    Write-Host "      $IconPath" -ForegroundColor Yellow
    # Create a minimal valid .ico from a PowerShell-generated PNG
    try {
        Add-Type -AssemblyName System.Drawing
        $bmp = New-Object System.Drawing.Bitmap(64, 64)
        $g = [System.Drawing.Graphics]::FromImage($bmp)
        $g.SmoothingMode = "HighQuality"
        # Draw a rocket icon (simple)
        $brush1 = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 69, 0))  # #FF4500
        $brush2 = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::White)
        $pen = New-Object System.Drawing.Pen([System.Drawing.Color]::White, 2)
        # Rocket body
        $g.FillEllipse($brush1, 20, 8, 24, 40)
        $g.FillRectangle($brush1, 16, 30, 32, 12)
        # Window
        $g.FillEllipse($brush2, 26, 18, 12, 12)
        # Flame
        $g.FillPolygon($brush1, @(
            [System.Drawing.Point]::new(20, 48),
            [System.Drawing.Point]::new(32, 56),
            [System.Drawing.Point]::new(44, 48)
        ))
        # Wings
        $g.FillPolygon($brush1, @(
            [System.Drawing.Point]::new(12, 30),
            [System.Drawing.Point]::new(16, 38),
            [System.Drawing.Point]::new(16, 30)
        ))
        $g.FillPolygon($brush1, @(
            [System.Drawing.Point]::new(52, 30),
            [System.Drawing.Point]::new(48, 38),
            [System.Drawing.Point]::new(48, 30)
        ))
        $g.Dispose()
        # Convert to icon
        $hIcon = $bmp.GetHicon()
        $icon = [System.Drawing.Icon]::FromHandle($hIcon)
        $fs = New-Object System.IO.FileStream($IconPath, [System.IO.FileMode]::Create)
        $icon.Save($fs)
        $fs.Close()
        $icon.Dispose()
        $bmp.Dispose()
        Write-Host "      [OK] icon.ico generado" -ForegroundColor Green
    }
    catch {
        Write-Host "      [WARN] No se pudo generar el icono: $_" -ForegroundColor Yellow
    }
}
else {
    Write-Host "      [OK] icon.ico encontrado" -ForegroundColor Green
}

<# Step 3: Find Inno Setup Compiler #>
Write-Host "[3/4] Buscando Inno Setup Compiler (iscc)..." -ForegroundColor Yellow
$IsccPaths = @(
    "${env:ProgramFiles(x86)}\Inno Setup 6\iscc.exe",
    "${env:ProgramFiles(x86)}\Inno Setup 5\iscc.exe",
    "${env:ProgramFiles}\Inno Setup 6\iscc.exe",
    "${env:ProgramFiles}\Inno Setup 5\iscc.exe",
    "${env:LOCALAPPDATA}\Programs\Inno Setup 6\iscc.exe",
    (Get-Command "iscc" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source)
)

$IsccPath = $null
foreach ($p in $IsccPaths) {
    if ($p -and (Test-Path $p)) {
        $IsccPath = $p
        break
    }
}

if (-not $IsccPath) {
    Write-Host "      [ERROR] Inno Setup Compiler no encontrado." -ForegroundColor Red
    Write-Host "      Instalelo desde: https://jrsoftware.org/isdl.php" -ForegroundColor Yellow
    Write-Host "      Luego ejecute este script nuevamente." -ForegroundColor Yellow
    exit 1
}
Write-Host "      [OK] iscc encontrado: $IsccPath" -ForegroundColor Green

<# Step 4: Build the installer #>
Write-Host "[4/4] Compilando instalador..." -ForegroundColor Yellow
$IssFile = Join-Path $ScriptDir "installer.iss"
$IsccArgs = @()
if ($LzmaThreads -gt 0) {
    # Inno Setup 6+ supports /O- for output override, we pass via #define
    $IsccArgs += "/dLzmaThreads=$LzmaThreads"
}
$IsccArgs += $IssFile

Write-Host "      Ejecutando: iscc $IsccArgs" -ForegroundColor Gray
Set-Location $ScriptDir

$proc = Start-Process -FilePath $IsccPath -ArgumentList $IsccArgs -NoNewWindow -Wait -PassThru
if ($proc.ExitCode -eq 0) {
    Write-Host ""
    Write-Host "╔══════════════════════════════════════════════════╗" -ForegroundColor Green
    Write-Host "║   [OK] Instalador compilado exitosamente!        ║" -ForegroundColor Green
    Write-Host "╚══════════════════════════════════════════════════╝" -ForegroundColor Green
    Write-Host ""
    Write-Host "Archivos generados:" -ForegroundColor White
    Get-ChildItem $OutputDir -Filter "*.exe" | ForEach-Object {
        $sizeInMB = [math]::Round($_.Length / 1MB, 2)
        Write-Host "  - $($_.Name) ($sizeInMB MB)" -ForegroundColor Green
        Write-Host "    Ruta: $($_.FullName)" -ForegroundColor Gray
    }
}
else {
    Write-Host "      [ERROR] La compilacion fallo con codigo: $($proc.ExitCode)" -ForegroundColor Red
    Write-Host "      Revise los mensajes de error de iscc arriba." -ForegroundColor Yellow
    exit $proc.ExitCode
}
