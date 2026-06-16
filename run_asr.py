"""
ASR runner — AudioService venv içinde çalışır.
Kullanım: python run_asr.py <wav_dosyasi> <cikti_txt>
Çıktı: transcript metni dosyaya yazılır
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"C:\Users\ilhami mert yeşilöz\Desktop\Ses\AudioService")))

from pipeline import asr as asr_module

wav_path = Path(sys.argv[1])
out_txt = Path(sys.argv[2])

asr_module.load_model({
    "models": {
        "whisper_size": "large-v3",
        "whisper_compute_type": "int8",
    }
})

segments = asr_module.transcribe(wav_path, language="tr")
text = " ".join(s.get("text", "").strip() for s in segments if s.get("text", "").strip())

out_txt.write_text(text, encoding="utf-8")
