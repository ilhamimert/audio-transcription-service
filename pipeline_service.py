"""
Unified Pipeline Servisi — tek process, modeller RAM'de sabit kalır
- Watchdog ile OS file event (polling yok)
- ASR (faster-whisper) + TTS (XTTS v2) aynı venv
- Modeller bir kez yüklenir, sonraki dosyalar saniyeler içinde işlenir
- done\ klasörü, kalıcı log dosyası, çoklu voice desteği

Kullanım:
  python pipeline_service.py --voice voices/emin.wav [--lang tr]
  python pipeline_service.py --voice voices/ [--lang tr]   # klasördeki ilk .wav
"""

import argparse
import logging
import os
import sys
import time
import queue
import threading
import subprocess
import psutil
from pathlib import Path

# ── CPU Optimizasyon ─────────────────────────────────────────────────────────
def _optimize_cpu():
    import torch

    phys_cores = psutil.cpu_count(logical=False) or 4
    logic_cores = psutil.cpu_count(logical=True) or 8

    torch.set_num_threads(phys_cores)
    torch.set_num_interop_threads(max(1, phys_cores // 2))

    os.environ["OMP_NUM_THREADS"]       = str(phys_cores)
    os.environ["MKL_NUM_THREADS"]       = str(phys_cores)
    os.environ["OPENBLAS_NUM_THREADS"]  = str(phys_cores)

    try:
        proc = psutil.Process()
        proc.nice(psutil.HIGH_PRIORITY_CLASS)
        phys_ids = list(range(0, logic_cores, logic_cores // phys_cores))
        proc.cpu_affinity(phys_ids)
        return phys_cores, phys_ids
    except Exception:
        return phys_cores, []


# ── Paths ────────────────────────────────────────────────────────────────────
FFMPEG = r"C:\Users\ILHAMI~1\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe"
BASE_DIR       = Path(r"D:\Masaüstü\AudioService")
INPUT_DIR      = BASE_DIR / "input"
PROCESSING_DIR = BASE_DIR / "processing"
TTS_OUTPUT_DIR = BASE_DIR / "tts_output"
DONE_DIR       = BASE_DIR / "done"
FAILED_DIR     = BASE_DIR / "failed"
LOGS_DIR       = BASE_DIR / "logs"
VOICES_DIR     = BASE_DIR / "voices"
SUPPORTED      = {".wav", ".mp3", ".m4a", ".mp4"}
MAX_RETRY      = 3


# ── Logging ──────────────────────────────────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_logger: logging.Logger | None = None


def _setup_logging():
    global _logger
    LOGS_DIR.mkdir(exist_ok=True)
    log_file = LOGS_DIR / f"pipeline_{time.strftime('%Y-%m-%d')}.log"

    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)

    _logger = logging.getLogger("pipeline")
    _logger.setLevel(logging.DEBUG)
    _logger.addHandler(fh)
    _logger.addHandler(ch)


def log(msg: str):
    if _logger:
        _logger.info(msg)
    else:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Voice seçimi ──────────────────────────────────────────────────────────────
_current_voice: Path | None = None
_voice_lock = threading.Lock()


def resolve_voice(arg: str) -> Path:
    """--voice argümanı dosya veya klasör olabilir."""
    p = Path(arg)
    if p.is_file():
        return p
    if p.is_dir():
        wavs = sorted(p.glob("*.wav"))
        if not wavs:
            print(f"HATA: {p} klasöründe .wav dosyası yok")
            sys.exit(1)
        return wavs[0]
    # Göreli yol dene
    candidate = BASE_DIR / arg
    if candidate.is_file():
        return candidate
    if candidate.is_dir():
        wavs = sorted(candidate.glob("*.wav"))
        if wavs:
            return wavs[0]
    print(f"HATA: Ses örneği bulunamadı: {arg}")
    sys.exit(1)


def get_voice() -> Path:
    with _voice_lock:
        return _current_voice


def set_voice(path: Path):
    global _current_voice
    with _voice_lock:
        _current_voice = path
    log(f"Aktif ses: {path.name}")


# ── Patches (torch + torchaudio) ─────────────────────────────────────────────
def _apply_patches():
    import torch
    import soundfile as sf

    _orig = torch.load
    def _patched(*a, **kw):
        kw.setdefault("weights_only", False)
        return _orig(*a, **kw)
    torch.load = _patched

    import torchaudio
    def _sf_load(path, *args, **kwargs):
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(data.T), sr
    torchaudio.load = _sf_load


# ── Donanım algılama ─────────────────────────────────────────────────────────
def _detect_device() -> tuple[str, str]:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
            log(f"GPU algılandı: {name} ({vram} MB VRAM) → CUDA kullanılıyor")
            compute = "float16" if vram >= 4096 else "int8"
            return "cuda", compute
    except Exception:
        pass

    try:
        import torch
        if hasattr(torch, "hip") and torch.hip.is_available():
            log("AMD GPU algılandı (ROCm) → GPU kullanılıyor")
            return "cuda", "float16"
    except Exception:
        pass

    phys = psutil.cpu_count(logical=False) or 4
    log(f"GPU bulunamadı → CPU kullanılıyor ({phys} fiziksel çekirdek)")
    return "cpu", "int8"


# ── Model loading ─────────────────────────────────────────────────────────────
_asr_model = None
_tts_model  = None
_device     = "cpu"


def load_asr():
    global _asr_model, _device
    from faster_whisper import WhisperModel
    phys_cores = psutil.cpu_count(logical=False) or 4
    device, compute = _detect_device()
    _device = device

    kwargs = {"device": device, "compute_type": compute}
    if device == "cpu":
        kwargs["cpu_threads"] = phys_cores
        kwargs["num_workers"] = 1

    log(f"ASR modeli yükleniyor (large-v3 / {compute} / {device}) ...")
    _asr_model = WhisperModel("large-v3", **kwargs)
    log("ASR modeli hazır.")


def load_tts(lang: str):
    global _tts_model
    os.environ["COQUI_TOS_AGREED"] = "1"
    _apply_patches()
    from TTS.api import TTS
    gpu = _device == "cuda"
    log(f"TTS modeli yükleniyor (xtts_v2 / {'GPU' if gpu else 'CPU'}) ...")
    _tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=gpu)
    log("TTS modeli hazır.")


# ── Core pipeline steps ───────────────────────────────────────────────────────
def to_wav(src: Path) -> Path:
    dst = PROCESSING_DIR / (src.stem + "_conv.wav")
    r = subprocess.run(
        [FFMPEG, "-y", "-i", str(src),
         "-ar", "16000", "-ac", "1",
         "-af", "loudnorm",
         str(dst)],
        capture_output=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg: {r.stderr.decode(errors='replace')[-300:]}")
    return dst


def transcribe(wav: Path, lang: str) -> str:
    segments, info = _asr_model.transcribe(
        str(wav),
        beam_size=5,
        language=lang,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    text = " ".join(s.text.strip() for s in segments if s.text.strip())
    log(f"  ASR dil={info.language} | {len(text)} karakter")
    return text


def synthesize(text: str, voice_path: Path, out_path: Path, lang: str):
    import torch
    with torch.inference_mode():
        _tts_model.tts_to_file(
            text=text,
            speaker_wav=str(voice_path),
            language=lang,
            file_path=str(out_path),
        )


# ── File processor ────────────────────────────────────────────────────────────
def _move_to_done(src: Path):
    DONE_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    dst = DONE_DIR / f"{src.stem}_{ts}{src.suffix}"
    try:
        src.rename(dst)
        log(f"  Arşivlendi → done\\{dst.name}")
    except Exception as e:
        log(f"  Arşivleme hatası: {e}")


def _move_to_failed(src: Path, reason: str):
    FAILED_DIR.mkdir(exist_ok=True)
    dst = FAILED_DIR / src.name
    if dst.exists():
        dst.unlink()
    try:
        src.rename(dst)
    except Exception:
        pass
    log(f"  BAŞARISIZ → failed\\{src.name} | Sebep: {reason}")


def process(src: Path, lang: str):
    voice = get_voice()
    log(f"İşleniyor: {src.name} | Ses: {voice.name}")
    wav = None

    for attempt in range(1, MAX_RETRY + 1):
        try:
            PROCESSING_DIR.mkdir(exist_ok=True)

            if src.suffix.lower() in {".mp4", ".m4a", ".mp3"}:
                if attempt == 1:
                    log("  WAV'a dönüştürülüyor...")
                wav = to_wav(src)
            else:
                wav = src

            if attempt == 1:
                log("  Metne çevriliyor (ASR)...")
            text = transcribe(wav, lang)
            if not text:
                log("  UYARI: Metin boş, atlanıyor.")
                if wav and wav != src:
                    wav.unlink(missing_ok=True)
                _move_to_done(src)
                return

            log(f"  Metin: {text[:100]}{'...' if len(text) > 100 else ''}")

            TTS_OUTPUT_DIR.mkdir(exist_ok=True)
            out = TTS_OUTPUT_DIR / (src.stem + "_klon.wav")
            log(f"  Ses üretiliyor ({voice.name})...")
            synthesize(text, voice, out, lang)

            if wav and wav != src:
                wav.unlink(missing_ok=True)
            _move_to_done(src)
            log(f"  TAMAM → tts_output\\{out.name}")
            return

        except Exception as e:
            if wav and wav.exists() and wav != src:
                wav.unlink(missing_ok=True)
            wav = None

            if attempt < MAX_RETRY:
                log(f"  HATA (deneme {attempt}/{MAX_RETRY}): {e} — {5 * attempt}sn sonra tekrar...")
                time.sleep(5 * attempt)
            else:
                _move_to_failed(src, str(e)[:120])


# ── Voice değiştirme komutu (stdin) ──────────────────────────────────────────
def _voice_command_listener(lang: str):
    """
    Servis çalışırken terminale komut girilmesini dinler.
    'voice voices/baska.wav' yazınca aktif sesi değiştirir.
    'voices' yazınca mevcut .wav dosyalarını listeler.
    """
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "voices":
            wavs = list(VOICES_DIR.glob("*.wav")) if VOICES_DIR.exists() else []
            if wavs:
                log("Mevcut ses örnekleri:")
                for w in wavs:
                    marker = " ← aktif" if w == get_voice() else ""
                    log(f"  {w.name}{marker}")
            else:
                log("voices\\ klasörü boş.")
        elif line.startswith("voice "):
            new_voice = resolve_voice(line[6:].strip())
            set_voice(new_voice)
        else:
            log(f"Bilinmeyen komut: {line} | Komutlar: voices | voice <dosya>")


# ── Watchdog + queue ──────────────────────────────────────────────────────────
file_queue: queue.Queue = queue.Queue()


def _worker(lang: str):
    while True:
        src = file_queue.get()
        if src is None:
            break
        process(src, lang)
        file_queue.task_done()


def _start_watchdog():
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    class _Handler(FileSystemEventHandler):
        def on_created(self, event):
            if event.is_directory:
                return
            p = Path(event.src_path)
            if p.suffix.lower() not in SUPPORTED:
                return
            time.sleep(0.5)
            if not p.exists():
                return
            proc = PROCESSING_DIR / p.name
            PROCESSING_DIR.mkdir(exist_ok=True)
            if proc.exists():
                proc.unlink()
            try:
                p.rename(proc)
                file_queue.put(proc)
            except Exception as e:
                log(f"Taşıma hatası: {e}")

    obs = Observer()
    obs.schedule(_Handler(), str(INPUT_DIR), recursive=False)
    obs.start()
    return obs


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", required=True,
                        help="Ses örneği dosyası veya voices/ klasörü")
    parser.add_argument("--lang", default="tr")
    args = parser.parse_args()

    _setup_logging()

    voice = resolve_voice(args.voice)
    set_voice(voice)

    for d in (INPUT_DIR, TTS_OUTPUT_DIR, PROCESSING_DIR, FAILED_DIR, DONE_DIR, LOGS_DIR):
        d.mkdir(exist_ok=True)

    phys, affinity = _optimize_cpu()
    log(f"CPU: {phys} fiziksel çekirdek | affinity={affinity} | öncelik=Yüksek")

    load_asr()
    load_tts(args.lang)

    # Worker thread
    t = threading.Thread(target=_worker, args=(args.lang,), daemon=True)
    t.start()

    # Voice komut dinleyici (arka planda)
    cmd_thread = threading.Thread(
        target=_voice_command_listener, args=(args.lang,), daemon=True
    )
    cmd_thread.start()

    obs = _start_watchdog()
    log(f"Servis hazır. İzleniyor: {INPUT_DIR}")
    log(f"Aktif ses: {voice}")
    log("Komutlar: 'voices' → listeye bak | 'voice voices/baska.wav' → ses değiştir")
    log("Dosya bekleniyor...\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log("Durduruluyor...")
        obs.stop()
        file_queue.put(None)
    obs.join()


if __name__ == "__main__":
    main()
