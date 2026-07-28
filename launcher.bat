@echo off
rem ---------------------------------------------------------------------------
rem HYDRAS Live Sim - launcher standalone (Windows).
rem Scarica src\live_sim.py dal repo GitHub (URL FISSO: nessun git richiesto) e
rem lo avvia; da li' live_sim.py scarica il resto (requisiti, script, dati,
rem modelli). Richiede Python 3 + curl (incluso in Windows 10/11) o PowerShell.
rem Uso: doppio clic, oppure  launcher.bat
rem ---------------------------------------------------------------------------
setlocal enabledelayedexpansion

set "REPO=MattiaManneschi/HYDRAS-Project"
set "BRANCH=master"
set "RAW_URL=https://raw.githubusercontent.com/%REPO%/%BRANCH%/src/live_sim.py"

if not defined HYDRAS_HOME set "HYDRAS_HOME=%~dp0"
set "HOME_DIR=%HYDRAS_HOME%"
if not exist "%HOME_DIR%\src" mkdir "%HOME_DIR%\src"
set "TARGET=%HOME_DIR%\src\live_sim.py"

if not exist "%TARGET%" (
  echo Scarico live_sim.py...
  where curl >nul 2>nul
  if !errorlevel! equ 0 (
    curl -fSL -o "%TARGET%" "%RAW_URL%"
  ) else (
    powershell -NoProfile -Command "Invoke-WebRequest -Uri '%RAW_URL%' -OutFile '%TARGET%'"
  )
)

if not exist "%TARGET%" (
  echo Errore: download di live_sim.py fallito.
  exit /b 1
)

set "PY="
where python >nul 2>nul && set "PY=python"
if not defined PY ( where py >nul 2>nul && set "PY=py" )
if not defined PY (
  echo Errore: Python non e' installato. Installa Python 3 e riprova.
  exit /b 1
)

set "HYDRAS_HOME=%HOME_DIR%"
"%PY%" "%TARGET%" %*
