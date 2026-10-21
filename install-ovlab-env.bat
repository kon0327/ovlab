@echo off
rem Keep environment changes in this Command Prompt after the installer exits.
set "OVLAB_SETUP_SCRIPT=%~dp0deploy\scripts\setup_ovlab_env.py"
if defined OVLAB_BOOTSTRAP_PYTHON (
    "%OVLAB_BOOTSTRAP_PYTHON%" "%OVLAB_SETUP_SCRIPT%" %*
) else (
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3 "%OVLAB_SETUP_SCRIPT%" %*
    ) else (
        python "%OVLAB_SETUP_SCRIPT%" %*
    )
)
if errorlevel 1 exit /b %ERRORLEVEL%

set "OVLAB_ENV_SETUP_PREFIX=%~dp0.ovlab-env"
set "OVLAB_ENV_SETUP_PREFIX_SUPPLIED=0"
set "OVLAB_ENV_SETUP_MANAGER=auto"
:parse_arguments
if "%~1"=="" goto activate_environment
set "OVLAB_ENV_SETUP_ARG=%~1"
if /I "%OVLAB_ENV_SETUP_ARG:~0,9%"=="--prefix=" (
    set "OVLAB_ENV_SETUP_PREFIX=%OVLAB_ENV_SETUP_ARG:~9%"
    set "OVLAB_ENV_SETUP_PREFIX_SUPPLIED=1"
)
if /I "%OVLAB_ENV_SETUP_ARG:~0,10%"=="--manager=" set "OVLAB_ENV_SETUP_MANAGER=%OVLAB_ENV_SETUP_ARG:~10%"
if /I "%~1"=="--prefix" (
    set "OVLAB_ENV_SETUP_PREFIX=%~2"
    set "OVLAB_ENV_SETUP_PREFIX_SUPPLIED=1"
    shift
)
if /I "%~1"=="--manager" (
    set "OVLAB_ENV_SETUP_MANAGER=%~2"
    shift
)
shift
goto parse_arguments

:activate_environment
if /I "%OVLAB_ENV_SETUP_MANAGER%"=="auto" (
    where conda >nul 2>nul
    if errorlevel 1 (set "OVLAB_ENV_SETUP_MANAGER=venv") else (set "OVLAB_ENV_SETUP_MANAGER=conda")
)
if /I "%OVLAB_ENV_SETUP_MANAGER%"=="conda" if "%OVLAB_ENV_SETUP_PREFIX_SUPPLIED%"=="0" goto activate_named_conda
for %%I in ("%OVLAB_ENV_SETUP_PREFIX%") do set "OVLAB_ENV_SETUP_PREFIX=%%~fI"
if /I "%OVLAB_ENV_SETUP_MANAGER%"=="conda" (
    call conda activate "%OVLAB_ENV_SETUP_PREFIX%"
) else (
    call "%OVLAB_ENV_SETUP_PREFIX%\Scripts\activate.bat"
)
if errorlevel 1 exit /b %ERRORLEVEL%
set "OVLAB_ENV_SETUP_TARGET=%OVLAB_ENV_SETUP_PREFIX%"
goto activation_complete

:activate_named_conda
call conda activate ovlab-env
if errorlevel 1 exit /b %ERRORLEVEL%
set "OVLAB_ENV_SETUP_TARGET=ovlab-env"

:activation_complete
echo OVLAB environment active: %OVLAB_ENV_SETUP_TARGET%
set "OVLAB_SETUP_SCRIPT="
set "OVLAB_ENV_SETUP_PREFIX="
set "OVLAB_ENV_SETUP_PREFIX_SUPPLIED="
set "OVLAB_ENV_SETUP_MANAGER="
set "OVLAB_ENV_SETUP_ARG="
set "OVLAB_ENV_SETUP_TARGET="
ovlab --version
exit /b %ERRORLEVEL%
