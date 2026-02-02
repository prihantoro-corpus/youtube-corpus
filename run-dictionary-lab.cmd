@echo off
echo Starting Dictionary Lab...

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_dictionary_lab.ps1"

pause
