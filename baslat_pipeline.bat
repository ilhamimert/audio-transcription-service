@echo off
chcp 65001 >nul
title Ses Pipeline Servisi
echo ========================================
echo  Ses Pipeline Servisi Baslatiliyor
echo  Kaynak ses: voices\emin.wav
echo  input\ klasorune dosya birakin
echo  Cikti: tts_output\ klasoru
echo  Islenip biten dosyalar: done\ klasoru
echo  Hatali dosyalar: failed\ klasoru
echo  Log kayitlari: logs\ klasoru
echo ----------------------------------------
echo  Komutlar (servisi durdurma gerek yok):
echo    voices           - mevcut ses orneklerini listele
echo    voice voices\baska.wav  - aktif sesi degistir
echo ========================================
echo.

cd /d D:\Masaüstü\AudioService
D:\Masaüstü\AudioService\tts_venv\Scripts\python.exe pipeline_service.py --voice voices\emin.wav --lang tr

pause
