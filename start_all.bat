@echo off
echo Starting Ollama, WebUI Forge, and Docker services...

REM Start Ollama in a new window
start "Ollama Server" cmd /c "echo Starting Ollama Server... && ollama serve"

REM Wait a moment for Ollama to initialize
timeout /t 5

REM Start WebUI Forge in a new window
start "WebUI Forge" cmd /c "echo Starting WebUI Forge... && call %~dp0start_webui_forge.bat"

REM Wait a moment for WebUI to initialize
timeout /t 5

REM Start Docker services
echo Starting Docker services...
docker-compose up -d

echo All services started!
echo.
echo - Ollama is running locally on port 11434
echo - WebUI Forge is running according to its configuration
echo - Inference Manager is running on port 8000
echo - N8N is running on port 5678
echo.
echo Press any key to shut down all services...
pause

REM Shutdown sequence
echo Shutting down Docker services...
docker-compose down

echo.
echo Services have been shut down.
echo You may need to manually close the Ollama and WebUI Forge windows.
echo.
pause