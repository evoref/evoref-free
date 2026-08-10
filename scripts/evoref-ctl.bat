@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
cd /d "%PROJECT_ROOT%"

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

if "%~1"=="" goto usage
if /i "%~1"=="start" goto start
if /i "%~1"=="start-core" goto start_core
if /i "%~1"=="stop" goto stop
if /i "%~1"=="restart" goto restart
if /i "%~1"=="status" goto status
goto usage

rem ── サービス起動 (全サービス: llama + backend + frontend) ──
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

rem ── コアサービス起動 (llama + backend のみ、frontend は起動しない) ──
rem `:start` から call され、reset_local_data.py の再起動からは `start-core`
rem コマンドとして直接呼ばれる。frontend(vite:5173) を生かしたまま llama と
rem backend だけを再起動する用途。stop が効くようウィンドウタイトルは維持する。
:start_core
call :check_dep python "python.org or winget install Python.Python.3"
if errorlevel 1 exit /b 1

rem 仮想環境の有効化
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

rem venv の Python / uvicorn を直接参照（start cmd /c では activate が引き継がれない）
if exist ".venv\Scripts\python.exe" (
    set "VENV_PYTHON=%CD%\.venv\Scripts\python.exe"
    set "VENV_UVICORN=%CD%\.venv\Scripts\uvicorn.exe"
) else (
    set "VENV_PYTHON=python"
    set "VENV_UVICORN=uvicorn"
)

rem 旧 llama-server が port を握ったまま新プロセスが bind 失敗すると、
rem 新プロセスは即死するが旧プロセスの /health 応答を拾って誤って ready 判定
rem されてしまう (モデル切替が反映されない不具合の原因)。spawn 前に一掃する。
echo [start] Ensuring no stale llama-server.exe is running...
taskkill /im "llama-server.exe" /f >nul 2>&1

echo [start] Starting llama-server (base + embedding; assist starts on demand)...
start "llama-server" /min cmd /c ""%VENV_PYTHON%" scripts\launch_llama.py config.yaml --all"

rem 待ち受け対象は launch_llama.py に問い合わせる。assist は
rem assist_model.residency=on_demand (既定) だと --all でも起動しないため、
rem 固定リストで待つと 60 秒空振りしてから WARNING が出てしまう。
echo [start] Waiting for llama-server to be ready (up to 60s)...
powershell -NoProfile -Command "$pairs=& '%VENV_PYTHON%' scripts\launch_llama.py config.yaml --print-health-ports; foreach ($p in $pairs) { if ($p -notmatch '^(\w+)=(\d+)$') { continue }; $name=$Matches[1]; $port=$Matches[2]; $elapsed=0; do { Start-Sleep 2; $elapsed+=2; try { $r=(Invoke-WebRequest \"http://localhost:$port/health\" -TimeoutSec 1 -UseBasicParsing).StatusCode } catch { $r=0 } } while ($r -ne 200 -and $elapsed -lt 60); if ($r -ne 200) { Write-Host \"[start] WARNING: $name (port $port) health check timed out, proceeding anyway\" } }"

echo [start] Starting FastAPI backend on :8000...
start "evoref-backend" /min cmd /c ""%VENV_UVICORN%" backend.main:app --host 0.0.0.0 --port 8000"
goto :eof

rem ── サービス停止 ──
:stop
echo [stop] Stopping evoref services...

taskkill /fi "WINDOWTITLE eq llama-server" /t /f >nul 2>&1
taskkill /im "llama-server.exe" /f >nul 2>&1
echo   llama-server stopped

taskkill /fi "WINDOWTITLE eq evoref-backend" /t /f >nul 2>&1
echo   FastAPI backend stopped

taskkill /fi "WINDOWTITLE eq evoref-frontend" /t /f >nul 2>&1
echo   SvelteKit frontend stopped

echo.
echo === All services stopped ===
goto :eof

rem ── サービス再起動 ──
:restart
call :stop
timeout /t 2 /nobreak >nul
call :start
goto :eof

rem ── サービス状態表示 ──
:status
echo === evoref service status ===

rem llama-server: 実行ファイル名で確認
tasklist /fi "IMAGENAME eq llama-server.exe" /nh 2>nul | find /i "llama-server" >nul 2>&1
if !errorlevel!==0 (
    echo   llama-server:       running
) else (
    echo   llama-server:       stopped
)

rem FastAPI backend: ポート 8000 で確認
netstat -an 2>nul | find ":8000 " | find "LISTENING" >nul 2>&1
if !errorlevel!==0 (
    echo   FastAPI backend:    running
) else (
    echo   FastAPI backend:    stopped
)

rem SvelteKit frontend: ポート 5173 で確認
netstat -an 2>nul | find ":5173 " | find "LISTENING" >nul 2>&1
if !errorlevel!==0 (
    echo   SvelteKit frontend: running
) else (
    echo   SvelteKit frontend: stopped
)
goto :eof

rem ── ヘルプ表示 ──
:usage
echo Usage: %~nx0 {start^|stop^|restart^|status}
echo.
echo Commands:
echo   start    Start all services (llama-server, FastAPI, SvelteKit)
echo   stop     Stop all running services
echo   restart  Restart all services
echo   status   Show status of all services
exit /b 1

rem ── 依存コマンド確認 ──
:check_dep
where %~1 >nul 2>&1
if errorlevel 1 (
    echo ERROR: '%~1' is not installed.
    if not "%~2"=="" echo   Install: %~2
    exit /b 1
)
exit /b 0
