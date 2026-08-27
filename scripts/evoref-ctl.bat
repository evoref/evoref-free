@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
cd /d "%PROJECT_ROOT%"

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

if "%~1"=="" goto usage

rem Second argument selects the develop level and is handed to the backend via
rem EVOREF_DEVELOP_LEVEL (backend/factory/_bootstrap.py reads it). The backend is
rem spawned as `uvicorn backend.main:app`, bypassing the CLI, so a `--develop`
rem flag cannot reach it -- the environment variable is the only channel.
rem Without this, `evoref-ctl start` always ran at the normal level and the
rem debug JSONL under local/logs/debug stayed empty, which cost time in two live
rem audits (2026-08-23 and 2026-08-27) before the cause was spotted.
set "EVOREF_DEVELOP_LEVEL="
set "EVOREF_LEARNING_DISABLED="
if not "%~2"=="" call :parse_mode "%~2" || exit /b 1
if not "%~3"=="" call :parse_mode "%~3" || exit /b 1

if /i "%~1"=="start" goto start
if /i "%~1"=="start-core" goto start_core
if /i "%~1"=="stop" goto stop
if /i "%~1"=="restart" goto restart
if /i "%~1"=="status" goto status
goto usage

rem --- Parse one optional mode argument (develop level or --no-learning) ---
:parse_mode
set "ARG=%~1"
if /i "%ARG%"=="--no-learning" (
    set "EVOREF_LEARNING_DISABLED=1"
    echo [start] learning disabled ^(EVOREF_LEARNING_DISABLED=1^)
    exit /b 0
)
if /i "%ARG%"=="debug"       goto set_level
if /i "%ARG%"=="investigate" goto set_level
if /i "%ARG%"=="evolve"      goto set_level
if /i "%ARG:~0,10%"=="--develop=" (
    set "ARG=%ARG:~10%"
    if /i "!ARG!"=="debug"       goto set_level
    if /i "!ARG!"=="investigate" goto set_level
    if /i "!ARG!"=="evolve"      goto set_level
)
echo ERROR: unknown option '%~1' ^(expected debug ^| investigate ^| evolve ^| --no-learning^)
exit /b 1
:set_level
set "EVOREF_DEVELOP_LEVEL=%ARG%"
echo [start] develop level: %ARG% ^(EVOREF_DEVELOP_LEVEL^)
exit /b 0

rem --- Start all services (llama + backend + frontend) ---
:start
call :check_dep npm "nodejs.org or winget install OpenJS.NodeJS"
if errorlevel 1 exit /b 1

call :start_core
if errorlevel 1 exit /b 1

echo [start] Starting SvelteKit dev server on :5173...
start "evoref-frontend" /min cmd /c "cd frontend && npm run dev -- --host 0.0.0.0"

echo.
echo === evoref is running ===
echo   Web UI:   http://localhost:5173
echo   API:      http://localhost:8000
echo   llama:    http://localhost:8080 (base)
echo   embed:    http://localhost:8082 (embedding, if llama-cpp)
echo.
echo Close the terminal windows to stop services,
echo or run: %~nx0 stop
goto :eof

rem --- Start core services only (llama + backend; frontend is left running) ---
rem Called from `:start`, and invoked directly as the `start-core` command when
rem reset_local_data.py restarts services. Restarts llama and backend while
rem keeping frontend(vite:5173) alive. Window titles are kept so `stop` still works.
:start_core
call :check_dep python "python.org or winget install Python.Python.3"
if errorlevel 1 exit /b 1

rem Activate the virtual environment
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

rem Reference the venv python/uvicorn directly (activate is not inherited by start cmd /c)
if exist ".venv\Scripts\python.exe" (
    set "VENV_PYTHON=%CD%\.venv\Scripts\python.exe"
    set "VENV_UVICORN=%CD%\.venv\Scripts\uvicorn.exe"
) else (
    set "VENV_PYTHON=python"
    set "VENV_UVICORN=uvicorn"
)

rem If a stale llama-server still holds the port, the new process fails to bind
rem and dies at once, yet the old process keeps answering /health so we wrongly
rem judge it ready (this is why a model switch could appear to have no effect).
rem Sweep them out before spawning.
echo [start] Ensuring no stale llama-server.exe is running...
taskkill /im "llama-server.exe" /f >nul 2>&1

echo [start] Starting llama-server (base + embedding)...
start "llama-server" /min cmd /c ""%VENV_PYTHON%" scripts\launch_llama.py config.yaml --all"

rem wait targets come from launch_llama.py --print-health-ports
echo [start] Waiting for llama-server to be ready (up to 60s)...
powershell -NoProfile -Command "$pairs=& '%VENV_PYTHON%' scripts\launch_llama.py config.yaml --print-health-ports; foreach ($p in $pairs) { if ($p -notmatch '^(\w+)=(\d+)$') { continue }; $name=$Matches[1]; $port=$Matches[2]; $elapsed=0; do { Start-Sleep 2; $elapsed+=2; try { $r=(Invoke-WebRequest \"http://localhost:$port/health\" -TimeoutSec 1 -UseBasicParsing).StatusCode } catch { $r=0 } } while ($r -ne 200 -and $elapsed -lt 60); if ($r -ne 200) { Write-Host \"[start] WARNING: $name (port $port) health check timed out, proceeding anyway\" } }"

echo [start] Starting FastAPI backend on :8000...
start "evoref-backend" /min cmd /c ""%VENV_UVICORN%" backend.main:app --host 0.0.0.0 --port 8000"
goto :eof

rem --- Stop services ---
:stop
echo [stop] Stopping evoref services...

rem Window-title taskkill only matches processes started by this script.
rem Services started any other way (evoref serve / uvicorn / a wrapper) survive
rem it, and the old code printed "stopped" without checking. stop_services.py
rem kills by port occupancy and verifies the ports are actually free.
if exist ".venv\Scripts\python.exe" (
    set "STOP_PYTHON=%CD%\.venv\Scripts\python.exe"
) else (
    set "STOP_PYTHON=python"
)
"%STOP_PYTHON%" scripts\stop_services.py
if errorlevel 1 (
    echo.
    echo === Stop FAILED: some services are still listening ===
    exit /b 1
)

echo.
echo === All services stopped ===
goto :eof

rem --- Restart services ---
:restart
call :stop
timeout /t 2 /nobreak >nul
call :start
goto :eof

rem --- Show service status ---
:status
echo === evoref service status ===

rem llama-server: check by executable name
tasklist /fi "IMAGENAME eq llama-server.exe" /nh 2>nul | find /i "llama-server" >nul 2>&1
if !errorlevel!==0 (
    echo   llama-server:       running
) else (
    echo   llama-server:       stopped
)

rem FastAPI backend: check port 8000
netstat -an 2>nul | find ":8000 " | find "LISTENING" >nul 2>&1
if !errorlevel!==0 (
    echo   FastAPI backend:    running
) else (
    echo   FastAPI backend:    stopped
)

rem SvelteKit frontend: check port 5173
netstat -an 2>nul | find ":5173 " | find "LISTENING" >nul 2>&1
if !errorlevel!==0 (
    echo   SvelteKit frontend: running
) else (
    echo   SvelteKit frontend: stopped
)
goto :eof

rem --- Usage ---
:usage
echo Usage: %~nx0 {start^|stop^|restart^|status} [debug^|investigate^|evolve] [--no-learning]
echo.
echo Commands:
echo   start    Start all services (llama-server, FastAPI, SvelteKit)
echo   stop     Stop all running services
echo   restart  Restart all services
echo   status   Show status of all services
echo.
echo Options (start / start-core / restart):
echo   debug^|investigate^|evolve   Develop level; sets EVOREF_DEVELOP_LEVEL for the backend.
echo                              Without it the backend runs at the normal level and
echo                              local/logs/debug stays empty.
echo   --no-learning              Disable the self-learning cycle (EVOREF_LEARNING_DISABLED=1).
echo.
echo Examples:
echo   %~nx0 start evolve
echo   %~nx0 restart investigate --no-learning
exit /b 1

rem --- Dependency check ---
:check_dep
where %~1 >nul 2>&1
if errorlevel 1 (
    echo ERROR: '%~1' is not installed.
    if not "%~2"=="" echo   Install: %~2
    exit /b 1
)
exit /b 0
