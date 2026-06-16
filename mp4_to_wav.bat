@echo off
if "%~1"=="" (
    echo Kullanim: Bir mp4 dosyasini bu bat dosyasinin uzerine suruklep birakin
    pause
    exit /b
)

set FFMPEG="C:\Users\ILHAMI~1\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe"
set INPUT=%~1
set OUTPUT=%~dp1%~n1.wav

echo Donusturuluyor: %INPUT%
echo Cikti: %OUTPUT%

%FFMPEG% -y -i "%INPUT%" -ar 16000 -ac 1 "%OUTPUT%"

if %errorlevel%==0 (
    echo.
    echo TAMAMLANDI: %OUTPUT%
) else (
    echo HATA olustu!
)
pause
