@echo off
setlocal
rem Windows wrapper. This file must use CRLF line endings: cmd.exe misparses
rem a .cmd file that was saved with Unix (LF-only) line endings and reports
rem "The filename, directory name, or volume label syntax is incorrect."
cd /d "%~dp0" || goto :nodir
set "PY=python"
py -3 -c "import sys" >nul 2>nul && set "PY=py -3"
%PY% scripts\bootstrap.py session %*
exit /b %errorlevel%

:nodir
echo Could not change to the project folder: "%~dp0"
echo If the project is on a network share, map it to a drive letter first.
exit /b 1
