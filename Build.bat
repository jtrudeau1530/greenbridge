@echo off
rem Double-click to build. Finds your copy of the game, checks what's installed, and builds into
rem Documents\Leonida. Extra options pass through, e.g.  Build.bat --out D:\Games\Leonida
setlocal
cd /d "%~dp0"
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY (
  echo Python 3.12 or newer is needed: https://www.python.org/downloads/
  echo Tick "Add python.exe to PATH" during setup, then run Build.bat again.
  pause
  exit /b 1
)
%PY% -I "%~dp0make_game.py" %*
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (echo Done.) else (echo The build stopped. Read the message above.)
pause
exit /b %RC%
