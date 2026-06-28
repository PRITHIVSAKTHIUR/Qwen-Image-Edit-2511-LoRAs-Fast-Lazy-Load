@echo off
chcp 65001 >nul
title Qwen-Image-Edit (Desktop)

cd /d "%~dp0"

echo.
echo [93m╔══════════════════════════════════════════════════╗[0m
echo [93m║    Qwen-Image-Edit - Desktop Native Window      ║[0m
echo [93m╚══════════════════════════════════════════════════╝[0m
echo.

:: Check if uv is available
where uv >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [91m[ERROR] uv no encontrado.[0m
    echo.
    echo Instalalo con:
    echo   powershell -c "irm https://astral.sh/uv/install.ps1 ^| iex"
    echo.
    pause
    exit /b 1
)

:: Check if virtual env exists and is synced
if not exist ".venv" (
    echo [93m[..] Primera ejecucion - instalando dependencias...[0m
    echo [93m[..] Esto puede tomar varios minutos (PyTorch ~3GB).[0m
    echo.
    uv sync
    if %ERRORLEVEL% neq 0 (
        echo [91m[ERROR] Fallo al instalar dependencias.[0m
        pause
        exit /b 1
    )
    echo [92m[OK] Dependencias instaladas.[0m
    echo.
)

:: Ensure pywebview is installed
uv run python -c "import webview" 2>nul
if %ERRORLEVEL% neq 0 (
    echo [93m[..] Instalando pywebview para modo desktop...[0m
    uv add pywebview
    if %ERRORLEVEL% neq 0 (
        echo [91m[ERROR] Fallo al instalar pywebview.[0m
        pause
        exit /b 1
    )
)

echo [92m[OK] Iniciando aplicacion en ventana nativa...[0m
echo.
echo [90mCierra la ventana para detener la aplicacion.[0m
echo.

uv run app.py --desktop

if %ERRORLEVEL% neq 0 (
    echo.
    echo [91m[ERROR] La aplicacion termino con error.[0m
    pause
)
