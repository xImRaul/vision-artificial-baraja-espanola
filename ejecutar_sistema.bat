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
echo   [1] Ejecutar Sistema Principal (Foto / Camara en Vivo / Carpeta)
echo   [2] Calibrar Tapete Verde (0_calibrar_tapete.py)
echo   [3] Calibrar Cartas haciendo Clic (1_calibrar_cartas.py)
echo   [4] Evaluacion Cuantitativa + Figuras Memoria (3_evaluacion_cuantitativa.py)
echo   [5] Generar Banco de Escenas Sinteticas Demo (4_generar_escenas_demo.py)
echo   [0] Salir
echo =====================================================================
set /p OPC="Selecciona una opcion [1-5 / 0]: "

if "%OPC%"=="1" "%PY_EXE%" 2_main.py & pause & goto MENU
if "%OPC%"=="2" "%PY_EXE%" 0_calibrar_tapete.py & pause & goto MENU
if "%OPC%"=="3" "%PY_EXE%" 1_calibrar_cartas.py & pause & goto MENU
if "%OPC%"=="4" "%PY_EXE%" 3_evaluacion_cuantitativa.py & pause & goto MENU
if "%OPC%"=="5" "%PY_EXE%" 4_generar_escenas_demo.py & pause & goto MENU
if "%OPC%"=="0" exit /b 0
goto MENU
