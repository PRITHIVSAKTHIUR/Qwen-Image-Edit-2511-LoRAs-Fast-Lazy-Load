<#
.SYNOPSIS
    Checks Python availability for Qwen-Image-Edit and suggests fixes.
.DESCRIPTION
    Verifies:
      - Python 3.12+ is installed
      - uv is available (bundled or system)
      - CUDA-capable GPU is present
    Returns JSON output for the installer to parse.
#>

param(
    [string]$UvPath = ""
)

$result = @{
    python_ok = $false
    python_version = ""
    python_path = ""
    uv_ok = $false
    uv_path = ""
    cuda_ok = $false
    cuda_version = ""
    gpu_name = ""
    vram_mb = 0
    errors = @()
    warnings = @()
}

# 1. Check Python
try {
    $py = Get-Command "python" -ErrorAction Stop
    $result.python_path = $py.Source
    $ver = & python --version 2>&1
    $result.python_version = "$ver"
    if ($ver -match 'Python (\d+)\.(\d+)') {
        $major = [int]$Matches[1]
        $minor = [int]$Matches[2]
        if ($major -ge 3 -and $minor -ge 12) {
            $result.python_ok = $true
        } else {
            $result.warnings += "Python $major.$minor detectado. Se requiere Python 3.12+."
        }
    }
} catch {
    $result.errors += "Python no encontrado. Instale Python 3.12+ desde python.org"
}

# 2. Check uv
if ($UvPath -and (Test-Path $UvPath)) {
    $result.uv_ok = $true
    $result.uv_path = $UvPath
} else {
    try {
        $uv = Get-Command "uv" -ErrorAction Stop
        $result.uv_ok = $true
        $result.uv_path = $uv.Source
    } catch {
        $result.errors += "uv no encontrado. El instalador deberia incluir uv.exe."
    }
}

# 3. Check CUDA / GPU
try {
    $gpu = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -like '*NVIDIA*' } | Select-Object -First 1
    if ($gpu) {
        $result.gpu_name = $gpu.Name
        $result.vram_mb = [math]::Round($gpu.AdapterRAM / 1MB, 0)
        if ($result.vram_mb -ge 6144) {
            $result.cuda_ok = $true
        } else {
            $result.warnings += "VRAM: $($result.vram_mb) MB. Minimo recomendado: 6144 MB (6 GB)."
        }
        # Check driver / CUDA version via nvidia-smi
        try {
            $nvidiaSmi = Get-Command "nvidia-smi" -ErrorAction Stop
            $smiOut = & $nvidiaSmi --query-gpu=driver_version --format=csv,noheader 2>&1
            if ($smiOut -match '(\d+\.\d+)') {
                $result.cuda_version = "Driver: $($Matches[1])"
            }
        } catch {}
    } else {
        $result.warnings += "No se detecto GPU NVIDIA. Se requiere GPU con CUDA para ejecucion local."
    }
} catch {
    $result.warnings += "No se pudo verificar la GPU."
}

# Output as JSON
$result | ConvertTo-Json -Compress
