@echo off
rem Native Windows setup, called by Pixi through cmd.exe. No PowerShell required.
setlocal EnableExtensions DisableDelayedExpansion
if not defined PIXI_PROJECT_ROOT goto :not_in_pixi
if not defined CONDA_PREFIX goto :not_in_pixi

set "ACADOS_ROOT=%PIXI_PROJECT_ROOT%\acados"
set "ACADOS_BUILD=%ACADOS_ROOT%\build-windows"
set "ACADOS_PYTHON=%CONDA_PREFIX%\python.exe"
set "ACADOS_INTERFACE=%ACADOS_ROOT%\interfaces\acados_template"
set "ACADOS_RENDERER=%ACADOS_ROOT%\bin\t_renderer.exe"
set "ACADOS_RENDERER_URL=https://github.com/acados/tera_renderer/releases/download/v0.2.0/t_renderer-v0.2.0-windows-amd64.exe"

if not exist "%ACADOS_PYTHON%" (
    echo [Setup Acados] ERROR: Pixi Python not found. 1>&2
    goto :failed
)
for %%T in (git cmake gcc g++ mingw32-make) do (
    where %%T >nul 2>&1
    if errorlevel 1 (
        echo [Setup Acados] ERROR: %%T must be on PATH. 1>&2
        goto :failed
    )
)
"%ACADOS_PYTHON%" -c "import struct, subprocess; assert struct.calcsize('P') == 8, '64-bit Python required'; assert subprocess.check_output(['gcc', '-dumpmachine'], text=True).strip() == 'x86_64-w64-mingw32', 'x86-64 MinGW-w64 GCC required'"
if errorlevel 1 goto :failed

if exist "%ACADOS_ROOT%" goto :check_checkout
echo [Setup Acados] Cloning v0.5.1...
git clone --branch v0.5.1 https://github.com/acados/acados.git "%ACADOS_ROOT%"
if errorlevel 1 goto :failed

:check_checkout
rem Never reset or overwrite an existing checkout.
if not exist "%ACADOS_ROOT%\.git" (
    echo [Setup Acados] ERROR: Existing acados directory is not a Git checkout. 1>&2
    goto :failed
)
"%ACADOS_PYTHON%" -c "import subprocess, sys; tag = subprocess.check_output(['git', '-C', sys.argv[1], 'describe', '--tags', '--exact-match', 'HEAD'], text=True).strip(); assert tag == 'v0.5.1', 'Expected acados v0.5.1; existing checkout was left unchanged'" "%ACADOS_ROOT%"
if errorlevel 1 goto :failed

for %%L in (acados hpipm blasfeo) do if not exist "%ACADOS_ROOT%\bin\%%L.dll" goto :build
goto :install_interface

:build
echo [Setup Acados] Building Windows shared libraries...
git -C "%ACADOS_ROOT%" submodule update --init --recursive
if errorlevel 1 goto :failed
rem Python needs shared libraries. The example MPC uses HPIPM, not qpOASES.
cmake -S "%ACADOS_ROOT%" -B "%ACADOS_BUILD%" ^
    -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release ^
    "-DACADOS_INSTALL_DIR=%ACADOS_ROOT%" ^
    -DBUILD_SHARED_LIBS=ON -DBLASFEO_TARGET=GENERIC ^
    -DHPIPM_TARGET=GENERIC -DACADOS_WITH_QPOASES=OFF
if errorlevel 1 goto :failed
if not defined NUMBER_OF_PROCESSORS set "NUMBER_OF_PROCESSORS=2"
cmake --build "%ACADOS_BUILD%" --parallel %NUMBER_OF_PROCESSORS%
if errorlevel 1 goto :failed
cmake --install "%ACADOS_BUILD%"
if errorlevel 1 goto :failed
for %%L in (acados hpipm blasfeo) do (
    if not exist "%ACADOS_ROOT%\bin\%%L.dll" (
        echo [Setup Acados] ERROR: %%L.dll was not installed. 1>&2
        goto :failed
    )
)

:install_interface
"%ACADOS_PYTHON%" -c "import importlib.util, pathlib, sys; s = importlib.util.find_spec('acados_template'); sys.exit(0 if s and pathlib.Path(s.origin).resolve().parent.parent == pathlib.Path(sys.argv[1]).resolve() else 1)" "%ACADOS_INTERFACE%"
if not errorlevel 1 goto :renderer
echo [Setup Acados] Installing the Python interface...
"%ACADOS_PYTHON%" -m pip install -e "%ACADOS_INTERFACE%"
if errorlevel 1 goto :failed

:renderer
if exist "%ACADOS_RENDERER%" goto :validate
echo [Setup Acados] Downloading Tera renderer...
rem Use Pixi's Python standard library; publish only a complete download.
"%ACADOS_PYTHON%" -c "import pathlib, sys, urllib.request; target = pathlib.Path(sys.argv[2]); tmp = target.with_suffix('.exe.download'); urllib.request.urlretrieve(sys.argv[1], tmp); tmp.replace(target)" "%ACADOS_RENDERER_URL%" "%ACADOS_RENDERER%"
if errorlevel 1 goto :failed

:validate
set "ACADOS_SOURCE_DIR=%ACADOS_ROOT%"
set "ACADOS_INSTALL_DIR=%ACADOS_ROOT%"
set "TERA_PATH=%ACADOS_RENDERER%"
set "PATH=%ACADOS_ROOT%\bin;%PATH%"
"%ACADOS_PYTHON%" -c "import ctypes, os; import acados_template; ctypes.WinDLL(os.path.join(os.environ['ACADOS_SOURCE_DIR'], 'bin', 'acados.dll'), winmode=0); print('[Setup Acados] Python interface and DLL loading OK.')"
if errorlevel 1 goto :failed
echo [Setup Acados] Ready. Windows MPC controllers must use the CMake builder.

rem Export only activation variables to the calling shell; discard setup variables.
endlocal & set "ACADOS_SOURCE_DIR=%ACADOS_SOURCE_DIR%" & set "ACADOS_INSTALL_DIR=%ACADOS_INSTALL_DIR%" & set "TERA_PATH=%TERA_PATH%" & set "PATH=%PATH%"
exit /b 0

:not_in_pixi
echo [Setup Acados] ERROR: Run inside the repository Pixi environment. 1>&2
:failed
echo [Setup Acados] Setup failed. See the error above. 1>&2
endlocal
exit /b 1
