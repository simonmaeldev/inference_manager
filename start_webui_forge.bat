@echo off
cd /d "C:\Users\ApprenTyr\Documents\StabilityMatrix-win-x64\Data\Packages\stable-diffusion-webui-forge"

REM Set Python path if needed (uncomment and modify if necessary)
set PYTHON=C:\Users\ApprenTyr\Documents\StabilityMatrix-win-x64\Data\Packages\stable-diffusion-webui-forge\venv\Scripts\python.exe

REM Call the original webui.bat with your parameters
call webui.bat --nowebui --cuda-malloc --api --gradio-allowed-path "C:\Users\ApprenTyr\Documents\StabilityMatrix-win-x64\Data\Images"

REM Keep window open if there was an error
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Execution failed with error code %ERRORLEVEL%
    pause
)