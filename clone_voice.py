"""
Ses klonlama scripti — XTTS v2
Kullanım:
  python clone_voice.py --voice voices/ornek.wav --text "Merhaba dünya" --out tts_output/cikti.wav
  python clone_voice.py --voice voices/ornek.wav --file metin.txt --out tts_output/cikti.wav
"""

import argparse
import os
import sys
from pathlib import Path

# Windows terminal encoding
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def synthesize(voice_path: str, text: str, out_path: str, lang: str = "tr"):
    import os
    import torch
    import soundfile as sf
    import numpy as np

    os.environ["COQUI_TOS_AGREED"] = "1"

    # torch 2.6+ weights_only=True varsayılan — TTS checkpoint'leri için False gerekli
    _original_torch_load = torch.load
    def _patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_load

    # torchaudio.load torchcodec kullanıyor — soundfile ile patch et
    import torchaudio
    def _patched_torchaudio_load(path, *args, **kwargs):
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        tensor = torch.from_numpy(data.T)
        return tensor, sr
    torchaudio.load = _patched_torchaudio_load

    from TTS.api import TTS
    print(f"Model yükleniyor: xtts_v2 ...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

    voice = Path(voice_path)
    if not voice.exists():
        print(f"HATA: Ses örneği bulunamadı: {voice_path}")
        sys.exit(1)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Ses klonlanıyor...")
    print(f"  Ses örneği : {voice_path}")
    print(f"  Metin      : {text[:80]}{'...' if len(text) > 80 else ''}")
    print(f"  Çıktı      : {out_path}")

    tts.tts_to_file(
        text=text,
        speaker_wav=str(voice),
        language=lang,
        file_path=str(out),
    )
    print(f"Tamamlandı: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="XTTS v2 ses klonlama")
    parser.add_argument("--voice", required=True, help="Ses örneği (.wav, 15-30 sn)")
    parser.add_argument("--text", help="Okunacak metin")
    parser.add_argument("--file", help="Okunacak metin dosyası (.txt)")
    parser.add_argument("--out", required=True, help="Çıktı ses dosyası (.wav)")
    parser.add_argument("--lang", default="tr", help="Dil kodu (varsayılan: tr)")
    args = parser.parse_args()

    if not args.text and not args.file:
        print("HATA: --text veya --file parametresi gerekli")
        sys.exit(1)

    if args.file:
        text = Path(args.file).read_text(encoding="utf-8").strip()
    else:
        text = args.text

    synthesize(args.voice, text, args.out, args.lang)


if __name__ == "__main__":
    main()
