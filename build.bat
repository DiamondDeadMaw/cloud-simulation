@echo off
:: build.bat - Script to compile the CUDA Raytracer

set CUDA_TOOLKIT_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"

set VCVARS_PATH="C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"


:: --- Environment Setup ---
echo Setting up MSVC environment using %VCVARS_PATH%...
call %VCVARS_PATH%
if %errorlevel% neq 0 (
    echo ERROR: Failed to initialize vcvars64.bat. Check VCVARS_PATH.
    goto :BuildFail
)
echo MSVC Environment variables set.

:: Check if CUDA Toolkit path exists
if not exist %CUDA_TOOLKIT_PATH%\bin\nvcc.exe (
    echo ERROR: CUDA Toolkit not found at %CUDA_TOOLKIT_PATH%. Please correct the CUDA_TOOLKIT_PATH variable in build.bat.
    goto :BuildFail
)

:: Explicitly add CUDA paths to the beginning of the PATH for this script's environment
echo Prepending CUDA paths to PATH environment variable...
set PATH=%CUDA_TOOLKIT_PATH%\bin;%CUDA_TOOLKIT_PATH%\libnvvp;%PATH%
echo Updated PATH (first few entries): %PATH:~0,200%...

:: Set CUDA_PATH (some tools might look for this)
set CUDA_PATH=%CUDA_TOOLKIT_PATH%
echo CUDA_PATH set to: %CUDA_PATH%

:: --- Diagnostics ---
echo.
echo Running Diagnostics:
echo   Looking for nvcc:
where nvcc
echo   NVCC Version:
nvcc --version
echo.
echo Environment appears set up. Starting NVCC build...
echo.

:: --- Run NVCC Build ---
:: %~dp0 expands to the directory containing this batch file
nvcc --expt-relaxed-constexpr cloud_kernel.cu cloud_simulation.cu cloud_sim.cpp glad.c -DGLFW_DLL -I "%~dp0include" -L "%~dp0lib" -Xcompiler "/openmp" -o "%~dp0cloud_sim_2.exe" -lglfw3dll -lopengl32 -lgdi32 -Xcudafe "--diag_suppress=20014"

:: --- Check Result ---
if %errorlevel% neq 0 (
    echo ERROR: NVCC compilation/linking failed.
    goto :BuildFail
)

echo Build successful! Output: %~dp0cloud_sim.exe
goto :BuildSuccess

:BuildFail
echo.
echo ********************
echo *** BUILD FAILED ***
echo ********************
exit /b 1

:BuildSuccess
echo.
echo **********************
echo *** BUILD COMPLETE ***
echo **********************
exit /b 0