@echo off
chcp 65001 >nul
title Qwen-Image-Edit

cd /d "%~dp0"

echo.
echo [93m╔══════════════════════════════════════════════════╗[0m
echo [93m║       Qwen-Image-Edit - Windows Launcher        ║[0m
echo [93m║        Powered by uv (https://astral.sh/uv)     ║[0m
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

echo [92m[OK] Iniciando aplicacion...[0m
echo.
echo  Abre http://localhost:7860 en tu navegador
echo.
echo [90mPresiona Ctrl+C en esta ventana para cerrar.[0m
echo.

uv run app.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [91m[ERROR] La aplicacion termino con error.[0m
    pause
)
