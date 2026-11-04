@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo =====================================================================
echo   SUBIR PROYECTO A TU REPOSITORIO DE GITHUB
echo =====================================================================
echo.
echo Pega la URL de tu repositorio de GitHub 
echo (Ejemplo: https://github.com/tu-usuario/vision-baraja-espanola.git)
echo.
set /p URL_REPO="URL de GitHub: "

if "%URL_REPO%"=="" (
    echo [ERROR] No has introducido ninguna URL.
    pause
    exit /b 1
)

git remote remove origin 2>nul
git remote add origin %URL_REPO%
git branch -M main

echo.
echo [INFO] Subiendo codigo y fotos a GitHub...
git push -u origin main

echo.
if %ERRORLEVEL% EQU 0 (
    echo =====================================================================
    echo   ¡SUBIDO CON EXITO A TU GITHUB!
    echo =====================================================================
) else (
    echo [AVISO] Si te ha pedido iniciar sesion, completa el login en el navegador.
)
pause
