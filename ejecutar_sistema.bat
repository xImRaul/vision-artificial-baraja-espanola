@echo off
chcp 65001 >nul
cd /d "%~dp0"

set "PY_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PY_EXE%" (
    if exist "%~dp0..\vision-baraja-espanola\.venv\Scripts\python.exe" (
        set "PY_EXE=%~dp0..\vision-baraja-espanola\.venv\Scripts\python.exe"
    ) else (
        echo [INFO] Creando entorno virtual .venv e instalando dependencias...
        python -m venv "%~dp0.venv" 2>nul || "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" -m venv "%~dp0.venv"
        "%PY_EXE%" -m pip install -r requirements.txt
    )
)

:MENU
cls
echo =====================================================================
echo   RETO 07 - SISTEMA EXPERTO DE VISION CLASICA (BARAJA ESPANOLA)
echo =====================================================================
echo   [1] Ejecutar Sistema Principal por Foto (2_main.py)
echo   [2] Calibrar Tapete Verde (calibrar_tapete.py)
echo   [3] Inspeccionar Cartas haciendo Clic (1-calibrar.py)
echo   [4] Evaluacion Cuantitativa + Matrices (3_evaluacion_cuantitativa.py)
echo   [5] Analizar Camara en Tiempo Real (2_main.py --camara 0)
echo   [0] Salir
echo =====================================================================
set /p OPC="Selecciona una opcion [1-5 / 0]: "

if "%OPC%"=="1" "%PY_EXE%" 2_main.py & pause & goto MENU
if "%OPC%"=="2" "%PY_EXE%" calibrar_tapete.py & pause & goto MENU
if "%OPC%"=="3" "%PY_EXE%" 1-calibrar.py & pause & goto MENU
if "%OPC%"=="4" "%PY_EXE%" 3_evaluacion_cuantitativa.py & pause & goto MENU
if "%OPC%"=="5" "%PY_EXE%" 2_main.py --camara 0 & pause & goto MENU
if "%OPC%"=="0" exit /b 0
goto MENU
