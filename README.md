# Audio Transcription & Voice Cloning Service

Ses dosyalarını otomatik olarak metne çeviren (ASR) ve seçilen ses örneğiyle yeniden seslendiren (TTS) bir pipeline servisi.

## Özellikler

- **ASR** — faster-whisper (large-v3) ile Türkçe/çok dilli transkripsiyon
- **TTS** — XTTS v2 ile ses klonlama
- **Watchdog** — `input/` klasörünü dinler, dosya gelince otomatik işler
- **CPU & GPU desteği** — CUDA varsa GPU, yoksa CPU'ya otomatik geçer
- **Çoklu ses desteği** — servis çalışırken ses örneği değiştirilebilir
- **Retry mekanizması** — başarısız dosyalar 3 kez tekrar denenir

## Klasör Yapısı

```
AudioService/
├── pipeline_service.py   # Ana servis (ASR + TTS pipeline)
├── clone_voice.py        # Tek seferlik ses klonlama scripti
├── run_asr.py            # Tek seferlik ASR scripti
├── baslat_pipeline.bat   # Servisi başlatan Windows bat dosyası
├── mp4_to_wav.bat        # MP4 → WAV dönüştürme yardımcısı
├── voices/               # Ses örnekleri (.wav, 15-30 sn önerilir)
├── input/                # İşlenecek dosyalar buraya bırakılır
├── processing/           # İşlem sırasında geçici klasör
├── tts_output/           # Klonlanmış ses çıktıları
├── done/                 # Başarıyla işlenen dosyalar
├── failed/               # Başarısız dosyalar
└── logs/                 # Günlük log dosyaları
```

## Kurulum

### Gereksinimler

- Python 3.10+
- ffmpeg (PATH'e ekli olmalı)
- CUDA (opsiyonel, GPU hızlandırma için)

### Paketler

```bash
pip install faster-whisper TTS watchdog psutil soundfile torch torchaudio
```

## Kullanım

### Pipeline Servisi (Önerilen)

Servisi başlatmak için bat dosyasını çalıştır:

```
baslat_pipeline.bat
```

veya direkt Python ile:

```bash
python pipeline_service.py --voice voices/ornek.wav --lang tr
```

Servis başladıktan sonra `input/` klasörüne `.wav`, `.mp3`, `.m4a` veya `.mp4` dosyası bırakın. Otomatik işlenir, çıktı `tts_output/` klasörüne kaydedilir.

**Çalışırken komutlar (servisi durdurmadan):**

```
voices                    → mevcut ses örneklerini listele
voice voices/baska.wav    → aktif sesi değiştir
```

### Tek Seferlik Ses Klonlama

```bash
python clone_voice.py --voice voices/ornek.wav --text "Merhaba dünya" --out tts_output/cikti.wav
python clone_voice.py --voice voices/ornek.wav --file metin.txt --out tts_output/cikti.wav --lang tr
```

### Tek Seferlik ASR (Transkripsiyon)

```bash
python run_asr.py ses_dosyasi.wav cikti.txt
```

## Desteklenen Formatlar

| Format | Destekleniyor |
|--------|---------------|
| .wav   | ✅ |
| .mp3   | ✅ |
| .m4a   | ✅ |
| .mp4   | ✅ |

## Notlar

- `voices/` klasöründeki ses örnekleri 15–30 saniye uzunluğunda, gürültüsüz olmalı
- Modeller ilk çalıştırmada `~/.local/share/tts/` altına indirilir
- GPU yoksa CPU modunda çalışır (daha yavaş)
- Log dosyaları `logs/pipeline_YYYY-MM-DD.log` formatında tutulur
