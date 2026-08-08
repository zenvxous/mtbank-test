"""Прогон test_data/ через реальный faster-whisper и расчёт WER против эталонов.

Печатает markdown-таблицу, готовую к вставке в README.

    uv run python scripts/eval_wer.py
    uv run python scripts/eval_wer.py --files call_dialog.wav card_block.mp3
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import jiwer
from num2words import num2words

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TEST_DATA = ROOT / "test_data"

SPEAKER_PREFIX = re.compile(r"^(Оператор|Клиент)\s*:\s*", re.MULTILINE)
NUMBER = re.compile(r"\d+")


def normalize(text: str) -> str:
    """Приводит эталон и гипотезу к сопоставимому виду.

    Whisper применяет inverse text normalization: произнесённое «десять тысяч»
    он пишет как «10 тысяч», а «процентов» — как «%». Эталон — это текст,
    который скармливался TTS, то есть словами. Без выравнивания этих форм WER
    показывает не ошибки распознавания, а разницу в форматировании.
    """
    text = text.lower().replace("ё", "е")
    text = text.replace("%", " процентов ")
    text = NUMBER.sub(lambda m: num2words(int(m.group()), lang="ru"), text)
    text = re.sub(r"[«»\"'—–\-.,!?;:()]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class Normalize(jiwer.AbstractTransform):
    def process_string(self, s: str) -> str:
        return normalize(s)


NORMALIZE = jiwer.Compose([Normalize(), jiwer.ReduceToListOfListOfWords()])
# Для CER та же нормализация, но сравниваем посимвольно: метрика устойчива
# к расхождениям в токенизации («мтбанк» / «мт банк», «email» / «e mail»).
NORMALIZE_CHARS = jiwer.Compose([Normalize(), jiwer.ReduceToListOfListOfChars()])


def load_reference(audio_path: Path) -> str:
    text = audio_path.with_suffix(".txt").read_text(encoding="utf-8")
    return SPEAKER_PREFIX.sub("", text).replace("\n", " ")


def probe(path: Path) -> tuple[str, int, float]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration:stream=codec_name,sample_rate",
            "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(result.stdout)
    stream = info["streams"][0]
    return stream["codec_name"], int(stream["sample_rate"]), float(info["format"]["duration"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", nargs="*", help="конкретные файлы вместо всех в test_data/")
    args = parser.parse_args()

    if args.files:
        audio_files = [TEST_DATA / name for name in args.files]
    else:
        audio_files = sorted(p for p in TEST_DATA.iterdir() if p.is_file() and p.suffix != ".md" and p.suffix != ".txt")

    missing = [p for p in audio_files if not p.with_suffix(".txt").exists()]
    if missing:
        print(f"Нет эталонов для: {', '.join(p.name for p in missing)}", file=sys.stderr)
        return 1

    from api.config import settings
    from asr.transcriber import AudioTranscriber

    print(f"Загружаю faster-whisper {settings.WHISPER_MODEL_SIZE} ({settings.WHISPER_DEVICE})...", file=sys.stderr)
    transcriber = AudioTranscriber()

    rows = []
    for path in audio_files:
        codec, sample_rate, duration = probe(path)
        print(f"ASR {path.name}...", file=sys.stderr)

        started = time.perf_counter()
        transcript = transcriber.transcribe(str(path))
        elapsed = time.perf_counter() - started

        reference = load_reference(path)
        hypothesis = " ".join(segment.text for segment in transcript.segments)
        words = jiwer.process_words(
            reference, hypothesis, reference_transform=NORMALIZE, hypothesis_transform=NORMALIZE
        )
        chars = jiwer.process_characters(
            reference, hypothesis, reference_transform=NORMALIZE_CHARS, hypothesis_transform=NORMALIZE_CHARS
        )
        rows.append(
            {
                "file": path.name,
                "codec": codec,
                "sample_rate": sample_rate,
                "duration": duration,
                "elapsed": elapsed,
                "wer": words.wer,
                "cer": chars.cer,
                "segments": len(transcript.segments),
            }
        )

    print()
    print("| Файл | Кодек | Sample rate | Длительность | Сегментов | Время ASR | RTF | WER | CER |")
    print("|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        print(
            f"| `{row['file']}` | {row['codec']} | {row['sample_rate']} Гц | {row['duration']:.0f} с | "
            f"{row['segments']} | {row['elapsed']:.0f} с | {row['elapsed'] / row['duration']:.2f} | "
            f"**{row['wer'] * 100:.1f}%** | {row['cer'] * 100:.1f}% |"
        )

    total_duration = sum(row["duration"] for row in rows)
    total_elapsed = sum(row["elapsed"] for row in rows)
    def weighted(key: str) -> float:
        """Средневзвешенная по длительности — иначе 5-секундный файл весит как трёхминутный."""
        return sum(row[key] * row["duration"] for row in rows) / total_duration

    print(
        f"| **Итого** | | | **{total_duration:.0f} с** | | **{total_elapsed:.0f} с** | "
        f"**{total_elapsed / total_duration:.2f}** | **{weighted('wer') * 100:.1f}%** | "
        f"**{weighted('cer') * 100:.1f}%** |"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
