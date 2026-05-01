@echo off
setlocal

REM Build Windows EXE for Audio Desktop App and copy it to Desktop.
cd /d "%~dp0"

echo [1/3] Installing/Updating PyInstaller...
py -m pip install --upgrade pyinstaller
if errorlevel 1 (
  echo Failed to install PyInstaller.
  exit /b 1
)

echo [2/3] Building SnapdragonAudioStudio.exe...
py -m PyInstaller --noconfirm --windowed --name SnapdragonAudioStudio --onefile audio_desktop_app.py
if errorlevel 1 (
  echo EXE build failed.
  exit /b 1
)

echo [3/3] Copying EXE to Desktop...
set "DESKTOP=C:\Users\look5\Desktop"
if not exist "%DESKTOP%" (
  echo Desktop path not found: %DESKTOP%
  echo Hint: edit build_windows_exe.bat and change DESKTOP if your username differs.
  exit /b 1
)

copy /Y "dist\SnapdragonAudioStudio.exe" "%DESKTOP%\SnapdragonAudioStudio.exe" >nul
if errorlevel 1 (
  echo Failed to copy EXE to Desktop.
  exit /b 1
)

echo Done. EXE is available at:
echo %DESKTOP%\SnapdragonAudioStudio.exe
exit /b 0
