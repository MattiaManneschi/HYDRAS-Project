@echo off
rem ---------------------------------------------------------------------------
rem HYDRAS Live Sim - standalone launcher (Windows).
rem Downloads src\live_sim.py from GitHub (FIXED URL: no git needed) and runs it;
rem live_sim.py then downloads the rest (requirements, scripts, data, models).
rem Requires Python 3 (with Tkinter) and curl (bundled in Windows 10/11) or
rem PowerShell. Usage: double-click, or run  launcher.bat
rem ---------------------------------------------------------------------------
setlocal enabledelayedexpansion

set "REPO=MattiaManneschi/HYDRAS-SourceSeeker"
set "BRANCH=master"
set "RAW_URL=https://raw.githubusercontent.com/%REPO%/%BRANCH%/src/live_sim.py"

if not defined HYDRAS_HOME set "HYDRAS_HOME=%~dp0"
set "HOME_DIR=%HYDRAS_HOME%"
if not exist "%HOME_DIR%\src" mkdir "%HOME_DIR%\src"
set "TARGET=%HOME_DIR%\src\live_sim.py"

if not exist "%TARGET%" (
  echo Downloading live_sim.py...
  where curl >nul 2>nul
  if !errorlevel! equ 0 (
    curl -fSL -o "%TARGET%" "%RAW_URL%"
  ) else (
    powershell -NoProfile -Command "Invoke-WebRequest -Uri '%RAW_URL%' -OutFile '%TARGET%'"
  )
)

if not exist "%TARGET%" (
  echo Error: failed to download live_sim.py.
  echo.
  pause
  exit /b 1
)

set "PY="
where python >nul 2>nul && set "PY=python"
if not defined PY ( where py >nul 2>nul && set "PY=py" )
if not defined PY (
  echo Error: Python is not installed. Install Python 3 and try again.
  echo.
  pause
  exit /b 1
)

set "HYDRAS_HOME=%HOME_DIR%"
"%PY%" "%TARGET%" %*
