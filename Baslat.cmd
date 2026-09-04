@echo off
setlocal

cd /d "%~dp0"
title Akilli PDF Arama Motoru

set "APP_URL=http://127.0.0.1:7860/"
set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"

rem Ayni uygulama zaten aciksa ikinci bir sunucu baslatma.
powershell.exe -NoLogo -NoProfile -NonInteractive -Command ^
  "try { $listener = Get-NetTCPConnection -LocalPort 7860 -State Listen -ErrorAction SilentlyContinue; if (-not $listener) { exit 1 }; $config = [string](Invoke-RestMethod -Uri 'http://127.0.0.1:7860/config' -TimeoutSec 2); if ($config -match 'app-phase3-styles' -and $config -match 'PDF Arama') { exit 0 }; exit 2 } catch { exit 2 }"

if errorlevel 2 goto port_conflict
if errorlevel 1 goto start_app

echo Uygulama zaten calisiyor. Tarayici aciliyor...
start "" "%APP_URL%"
exit /b 0

:port_conflict
echo.
echo HATA: 7860 portunu baska bir uygulama kullaniyor.
echo O uygulamayi kapatip Baslat.cmd dosyasini yeniden calistirin.
echo.
pause
exit /b 1

:start_app
if not exist "%PYTHON_EXE%" goto missing_venv
if not exist "%~dp0app.py" goto missing_app

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "GRADIO_SERVER_PORT=7860"
set "GRADIO_NUM_PORTS=1"

echo Akilli PDF Arama Motoru baslatiliyor...
echo Ilk acilis, modeller yuklenirken biraz surebilir.
echo Uygulama hazir olunca tarayici otomatik acilacak.
echo Bu pencereyi kapatmak uygulamayi da durdurur.
echo.

"%PYTHON_EXE%" -u app.py
set "APP_EXIT_CODE=%ERRORLEVEL%"

echo.
echo Uygulama durdu. Cikis kodu: %APP_EXIT_CODE%
pause
exit /b %APP_EXIT_CODE%

:missing_venv
echo.
echo HATA: Projenin sanal ortami bulunamadi:
echo %PYTHON_EXE%
echo venv klasorunu kurduktan sonra yeniden deneyin.
echo.
pause
exit /b 1

:missing_app
echo.
echo HATA: app.py proje klasorunde bulunamadi.
echo.
pause
exit /b 1
