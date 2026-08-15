@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
cd /d "%PROJECT_ROOT%"

rem --- Argument parsing ---
set "SHARED_PATH="
set "FORCE="
:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="--shared-path" (
    if "%~2"=="" (
        echo ERROR: --shared-path requires a path argument
        echo Usage: setup.bat [--shared-path ^<path^>] [--force]
        exit /b 1
    )
    set "SHARED_PATH=%~2"
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--force" (
    set "FORCE=1"
    shift
    goto parse_args
)
if /i "%~1"=="-h" goto show_help
if /i "%~1"=="--help" goto show_help
echo ERROR: Unknown option: %~1
echo Usage: setup.bat [--shared-path ^<path^>] [--force]
exit /b 1

:show_help
echo Usage: setup.bat [--shared-path ^<path^>] [--force]
echo.
echo Options:
echo   --shared-path ^<path^>  NAS shared path for multi-PC setup
echo                          Uses shared models/ (no model checks)
echo   --force                 Force reinstall (recreate .venv, reinstall packages,
echo                          overwrite config.yaml)
echo   -h, --help              Show this help message
exit /b 0

:args_done

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

rem --- Dependency check ---
call :check_dep python "python.org or winget install Python.Python.3"
if errorlevel 1 exit /b 1
call :check_dep npm "nodejs.org or winget install OpenJS.NodeJS"
if errorlevel 1 exit /b 1
call :check_dep git "git-scm.com or winget install Git.Git"
if errorlevel 1 exit /b 1

rem --- Model path validation ---
if defined SHARED_PATH (
    if not exist "!SHARED_PATH!" (
        echo ERROR: Shared path does not exist: !SHARED_PATH!
        exit /b 1
    )
    if not exist "!SHARED_PATH!\models" (
        echo ERROR: models\ directory not found in shared path: !SHARED_PATH!
        echo Expected structure:
        echo   !SHARED_PATH!\
        echo   +-- models\    ^(GGUF model files^)
        exit /b 1
    )
    rem Convert to an absolute path
    pushd "!SHARED_PATH!"
    set "SHARED_PATH=!CD!"
    popd
)

echo === evoref setup ===
echo Project root: %PROJECT_ROOT%
if defined SHARED_PATH echo Shared path:  !SHARED_PATH!
if defined FORCE echo Mode:          FORCE REINSTALL
echo.

rem --- 1. Python virtual environment ---
echo [1/6] Creating Python virtual environment...
if defined FORCE if exist ".venv" (
    echo   [force] Removing existing .venv...
    rmdir /s /q .venv
)
if not exist ".venv" (
    python -m venv .venv
    echo   Created .venv
) else (
    echo   .venv already exists, skipping
)

call .venv\Scripts\activate.bat

rem --- 2. Python packages ---
echo [2/6] Installing Python dependencies...
python -m pip install --upgrade pip -q
if defined FORCE (
    python -m pip install --force-reinstall -r backend\requirements.txt -q
    python -m pip install --force-reinstall -e . -q
) else (
    python -m pip install -r backend\requirements.txt -q
    python -m pip install -e . -q
)
echo   Done

rem --- 3. Frontend dependencies ---
echo [3/6] Installing frontend dependencies...
if defined FORCE if exist "frontend\node_modules" (
    echo   [force] Removing existing node_modules...
    rmdir /s /q frontend\node_modules
)
cd frontend
call npm install --silent
cd /d "%PROJECT_ROOT%"
echo   Done

rem --- 4. config.yaml ---
echo [4/6] Setting up config.yaml...
if defined FORCE if exist "config.yaml" (
    echo   [force] Overwriting existing config.yaml
    del config.yaml
)
if not exist "config.yaml" (
    if exist "config.yaml.example" (
        copy config.yaml.example config.yaml >nul
        echo   Copied config.yaml.example -^> config.yaml
    ) else (
        echo   WARNING: config.yaml.example not found. Please create config.yaml manually.
    )
) else (
    echo   config.yaml already exists, skipping
)

if defined SHARED_PATH if exist "config.yaml" (
    rem Look for a GGUF model file
    set "GGUF_PATH="
    for %%f in ("!SHARED_PATH!\models\*.gguf") do (
        if not defined GGUF_PATH (
            set "GGUF_PATH=%%f"
            echo   Found model: %%~nxf
        )
    )
    if not defined GGUF_PATH (
        set "GGUF_PATH=!SHARED_PATH!\models\gemma-4-12b-it-qat-q4_0.gguf"
        echo   WARNING: No .gguf file found in !SHARED_PATH!\models\
        echo            Please update model_paths.base_model in config.yaml manually.
    )
    rem Update model_paths in config.yaml
    python scripts\configure_shared_path.py "!SHARED_PATH!" "!GGUF_PATH!"
    echo   Updated config.yaml with shared paths
)

rem --- 5. Model placement check ---
echo [5/6] Checking models...
if defined SHARED_PATH (
    echo   Using shared path. Models expected at: !SHARED_PATH!\models\
) else (
    rem Automatic download was dropped; place GGUF files into models\ manually.
    python scripts\download_model.py
)

rem --- 6. Local directories ---
echo [6/6] Creating local directories...
for %%d in (
    "local\models"
    "local\vectors\chunks"
    "local\knowledge"
    "local\memory"
    "local\prompts\history"
    "local\cartridges"
    "local\history"
    "local\lora_versions"
    "local\logs\debug"
) do (
    if not exist %%d mkdir %%~d
)
echo   Done

echo.
echo === Setup complete ===
echo.
if defined SHARED_PATH (
    echo Shared path configured: !SHARED_PATH!
    echo.
    echo Next steps:
    echo   1. Install llama-server from https://github.com/ggml-org/llama.cpp/releases
    echo   2. Verify config.yaml settings ^(gpu_layers, context_size, etc.^)
    echo   3. Run: scripts\evoref-ctl.bat start
) else (
    echo Next steps:
    echo   1. Install llama-server from https://github.com/ggml-org/llama.cpp/releases
    echo   2. Edit config.yaml to set gpu_layers and context_size for your hardware
    echo   3. Run: scripts\evoref-ctl.bat start
)
echo.

endlocal
goto :eof

rem --- Dependency check ---
:check_dep
where %~1 >nul 2>&1
if errorlevel 1 (
    echo ERROR: '%~1' is not installed.
    if not "%~2"=="" echo   Install: %~2
    exit /b 1
)
exit /b 0
