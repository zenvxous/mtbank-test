"""Генерация тестовых аудиоданных в test_data/.

Синтезирует русские диалоги через edge-tts (голоса из docs/sample-dialog.md),
склеивает реплики с паузами через ffmpeg и раскладывает по разным форматам
и sample rate. Рядом с каждым аудио кладёт эталонный транскрипт .txt.

Отдельно генерирует негативные фикстуры (test_data/negative/) для проверки
обработки ошибок в api/main.py.

    uv run python scripts/make_test_data.py           # пропустить существующие
    uv run python scripts/make_test_data.py --force   # перегенерировать всё
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import shutil
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path

import edge_tts

ROOT = Path(__file__).resolve().parent.parent
TEST_DATA = ROOT / "test_data"
NEGATIVE = TEST_DATA / "negative"

OPERATOR = "Оператор"
CLIENT = "Клиент"

# Голоса ровно те, что предложены в docs/sample-dialog.md.
VOICES = {
    OPERATOR: "ru-RU-SvetlanaNeural",
    CLIENT: "ru-RU-DmitryNeural",
}

# Пауза между репликами, сек (docs/sample-dialog.md: 0.5–1 сек).
PAUSE_SECONDS = 0.6

# Промежуточный формат: все реплики и паузы приводятся к нему, чтобы
# ffmpeg concat-демуксер работал на однородных файлах.
INTERMEDIATE_ARGS = ["-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le"]


@dataclass
class Scenario:
    filename: str
    description: str
    # Кодек/rate/каналы целевого файла — передаются ffmpeg как есть.
    encode_args: list[str]
    lines: list[tuple[str, str]] = field(default_factory=list)


CALL_DIALOG_LINES: list[tuple[str, str]] = [
    (OPERATOR, "Добрый день, МТБанк, меня зовут Анна, чем могу помочь?"),
    (CLIENT, "Здравствуйте. Хочу узнать про условия по кредиту наличными."),
    (OPERATOR, "Конечно, подскажите, пожалуйста, какая сумма вас интересует и на какой срок?"),
    (CLIENT, "Примерно десять тысяч рублей, на год."),
    (
        OPERATOR,
        "Отлично. На данный момент ставка от четырнадцати и девяти процентов годовых, "
        "решение за пятнадцать минут. Вы уже являетесь клиентом МТБанка?",
    ),
    (CLIENT, "Да, у меня есть карточка ваша."),
    (
        OPERATOR,
        "Прекрасно, тогда для вас действуют специальные условия. Ежемесячный платёж составит "
        "около девятисот рублей. Вам удобно подать заявку онлайн через приложение или "
        "предпочитаете приехать в отделение?",
    ),
    (CLIENT, "Лучше онлайн. Но у меня вопрос — если я захочу досрочно погасить, есть штрафы?"),
    (OPERATOR, "Нет, досрочное погашение без штрафов и комиссий, в любое время и в любом объёме."),
    (CLIENT, "Хорошо, а страховка обязательна?"),
    (
        OPERATOR,
        "Страхование жизни подключается по вашему желанию, это не обязательное условие "
        "получения кредита. Однако при подключении страховки ставка может быть немного снижена.",
    ),
    (CLIENT, "Понятно. Тогда я попробую подать через приложение."),
    (
        OPERATOR,
        "Отлично. Если возникнут вопросы в процессе заполнения — звоните, мы поможем. "
        "Также могу отправить вам краткую инструкцию на email, если хотите.",
    ),
    (CLIENT, "Да, пожалуйста, отправьте."),
    (OPERATOR, "Хорошо, подскажите ваш email."),
    (CLIENT, "Михаил-собака-пример-точка-бай."),
    (
        OPERATOR,
        "Записала. В течение нескольких минут получите письмо с инструкцией и ссылкой на "
        "заявку. Есть ещё вопросы?",
    ),
    (CLIENT, "Нет, всё понятно, спасибо."),
    (OPERATOR, "Спасибо за обращение в МТБанк, хорошего дня!"),
    (CLIENT, "И вам, до свидания."),
]


SCENARIOS: list[Scenario] = [
    Scenario(
        filename="call_dialog.wav",
        description="Консультация по кредиту наличными, диалог оператора и клиента (docs/sample-dialog.md)",
        encode_args=["-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le"],
        lines=CALL_DIALOG_LINES,
    ),
    Scenario(
        filename="card_block.mp3",
        description="Клиент срочно блокирует утерянную карту, оператор оформляет перевыпуск",
        encode_args=["-ar", "44100", "-ac", "2", "-c:a", "libmp3lame", "-b:a", "128k"],
        lines=[
            (OPERATOR, "МТБанк, здравствуйте, меня зовут Ольга. Слушаю вас."),
            (CLIENT, "Здравствуйте! Я потерял карточку, надо срочно заблокировать."),
            (
                OPERATOR,
                "Понимаю, поможем прямо сейчас. Назовите, пожалуйста, вашу фамилию, имя "
                "и отчество и последние четыре цифры карты.",
            ),
            (CLIENT, "Ковалёв Сергей Петрович, карта заканчивается на семь два один девять."),
            (
                OPERATOR,
                "Спасибо. Карта заблокирована, операции по ней больше не пройдут. "
                "За последние сутки списаний, которые вы не совершали, я не вижу.",
            ),
            (CLIENT, "Слава богу. А новую карту как получить?"),
            (
                OPERATOR,
                "Перевыпуск я оформила, готовность три рабочих дня. Заберёте в том же отделении, "
                "где открывали счёт, реквизиты и остаток сохранятся.",
            ),
            (CLIENT, "Отлично, спасибо большое за помощь."),
            (OPERATOR, "Всегда рады помочь. Хорошего дня, до свидания."),
        ],
    ),
    Scenario(
        filename="deposit_question.ogg",
        description="Короткий вопрос по ставке вклада",
        encode_args=["-ar", "48000", "-ac", "1", "-c:a", "libopus", "-b:a", "32k"],
        lines=[
            (OPERATOR, "МТБанк, добрый день, меня зовут Ирина."),
            (CLIENT, "Добрый день. Подскажите, какая сейчас ставка по рублёвому вкладу на полгода?"),
            (
                OPERATOR,
                "На шесть месяцев ставка составляет одиннадцать процентов годовых, "
                "проценты выплачиваются в конце срока.",
            ),
            (CLIENT, "А пополнять можно?"),
            (OPERATOR, "Да, пополнение доступно в течение первых трёх месяцев."),
            (CLIENT, "Понял, спасибо."),
        ],
    ),
    Scenario(
        filename="complaint_violation.wav",
        description=(
            "Конфликтный звонок с нарушениями регламента: оператор не поздоровался и не "
            "представился, перебивает, обещает гарантированное одобрение, не прощается"
        ),
        encode_args=["-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le"],
        lines=[
            (OPERATOR, "Да, слушаю."),
            (
                CLIENT,
                "Здравствуйте, я уже третий раз звоню! У меня деньги списали дважды за одну "
                "покупку, и никто ничего не делает.",
            ),
            (OPERATOR, "Ну подождите, не надо на меня кричать. Сумма какая?"),
            (CLIENT, "Двести сорок рублей. Я вам вчера всё это уже рассказывал."),
            (
                OPERATOR,
                "Слушайте, я не знаю, что вам там вчера говорили. Ничего страшного не произошло, "
                "само вернётся.",
            ),
            (CLIENT, "Как само? Мне нужна заявка и номер обращения!"),
            (
                OPERATOR,
                "Да оформлю я вам заявку, не переживайте. Вам сто процентов всё вернут, "
                "гарантирую, даже не сомневайтесь.",
            ),
            (CLIENT, "А когда именно? И почему вы со мной так разговариваете?"),
            (OPERATOR, "Ну когда рассмотрят, тогда и вернут. Всё, у меня другие звонки."),
        ],
    ),
    Scenario(
        filename="short_greeting.mp3",
        description="Очень короткое обращение клиента, один говорящий",
        encode_args=["-ar", "22050", "-ac", "1", "-c:a", "libmp3lame", "-b:a", "64k"],
        lines=[
            (CLIENT, "Здравствуйте, я хотел бы уточнить баланс по своей карте."),
        ],
    ),
]

# Телефонное качество — производный файл, кодек μ-law 8 кГц (README: ffmpeg -ar 8000 -acodec pcm_mulaw).
TELEPHONE_SOURCE = "call_dialog.wav"
TELEPHONE_FILENAME = "call_dialog_8k.wav"
TELEPHONE_DESCRIPTION = "Тот же диалог, прогнанный через телефонный кодек μ-law 8 кГц"


def run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr[-2000:]}")


async def _synthesize(text: str, voice: str, out_path: Path) -> None:
    await edge_tts.Communicate(text, voice).save(str(out_path))


def synthesize_line(text: str, voice: str, out_path: Path) -> None:
    asyncio.run(_synthesize(text, voice, out_path))


def make_pause(out_path: Path, seconds: float) -> None:
    run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "anullsrc=channel_layout=mono:sample_rate=24000",
            "-t", str(seconds), *INTERMEDIATE_ARGS, str(out_path),
        ]
    )


def build_scenario(scenario: Scenario, workdir: Path) -> None:
    """Синтезирует реплики, склеивает их с паузами и кодирует в целевой формат."""
    parts: list[Path] = []
    pause = workdir / "pause.wav"
    make_pause(pause, PAUSE_SECONDS)

    for index, (speaker, text) in enumerate(scenario.lines):
        raw = workdir / f"line_{index:02d}.mp3"
        normalized = workdir / f"line_{index:02d}.wav"
        synthesize_line(text, VOICES[speaker], raw)
        run(["ffmpeg", "-y", "-i", str(raw), *INTERMEDIATE_ARGS, str(normalized)])
        if parts:
            parts.append(pause)
        parts.append(normalized)

    concat_list = workdir / "concat.txt"
    concat_list.write_text("".join(f"file '{part}'\n" for part in parts), encoding="utf-8")

    joined = workdir / "joined.wav"
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", str(joined)])
    run(["ffmpeg", "-y", "-i", str(joined), *scenario.encode_args, str(TEST_DATA / scenario.filename)])

    transcript = "\n".join(f"{speaker}: {text}" for speaker, text in scenario.lines)
    (TEST_DATA / scenario.filename).with_suffix(".txt").write_text(transcript + "\n", encoding="utf-8")


def build_telephone_copy() -> None:
    source = TEST_DATA / TELEPHONE_SOURCE
    target = TEST_DATA / TELEPHONE_FILENAME
    run(["ffmpeg", "-y", "-i", str(source), "-ar", "8000", "-ac", "1", "-acodec", "pcm_mulaw", str(target)])
    # Эталон копируем, а не симлинкуем: симлинки ломаются на Windows и при COPY в Docker.
    shutil.copyfile(source.with_suffix(".txt"), target.with_suffix(".txt"))


def build_negative_fixtures() -> None:
    """Битые/пограничные файлы для проверки валидации в api/main.py."""
    NEGATIVE.mkdir(parents=True, exist_ok=True)

    # 400 "Uploaded file is empty" — api/main.py
    (NEGATIVE / "empty.wav").write_bytes(b"")

    # 422 "Failed to decode audio" — настоящий ogg с испорченным началом
    # (оборвавшаяся закачка). Наивный вариант «WAV-заголовок + случайные байты»
    # не годится: случайные байты — валидный PCM, ffmpeg их декодирует как шум.
    # Портим детерминированно, чтобы файл не менялся при каждой перегенерации.
    source = (TEST_DATA / "deposit_question.ogg").read_bytes()
    noise = bytes(random.Random(20260808).getrandbits(8) for _ in range(4096))
    (NEGATIVE / "corrupted.ogg").write_bytes(noise + source[4096:])

    # 422 "File contains no audio stream" — расширение разрешённое, но внутри
    # контейнер только с видеодорожкой: проверка на то, что валидация не
    # ограничивается расширением имени файла.
    run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10",
            "-t", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
            "-f", "mp4", "-movflags", "+faststart", str(NEGATIVE / "no_audio_stream.wav"),
        ]
    )

    # 200 с пустым транскриптом: валидное аудио без речи.
    with wave.open(str(NEGATIVE / "silence_30s.wav"), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * (16000 * 30))

    # 400 "Unsupported file type: .txt"
    (NEGATIVE / "not_audio.txt").write_text(
        "Это не аудиофайл, а обычный текст — проверка валидации расширения.\n", encoding="utf-8"
    )


def probe(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration,format_name:stream=codec_name,sample_rate,channels",
            "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {}
    return json.loads(result.stdout)


def print_summary() -> None:
    print(f"\n{'файл':<28} {'формат':<12} {'кодек':<12} {'Гц':>7} {'кан':>4} {'сек':>7} {'КБ':>8}")
    print("-" * 84)
    total = 0.0
    for path in sorted(TEST_DATA.glob("*")):
        if path.suffix == ".txt" or path.is_dir():
            continue
        info = probe(path)
        stream = (info.get("streams") or [{}])[0]
        fmt = info.get("format", {})
        duration = float(fmt.get("duration", 0.0))
        total += duration
        print(
            f"{path.name:<28} {fmt.get('format_name', '?')[:12]:<12} {stream.get('codec_name', '?'):<12} "
            f"{stream.get('sample_rate', '?'):>7} {stream.get('channels', '?'):>4} "
            f"{duration:>7.1f} {path.stat().st_size / 1024:>8.0f}"
        )
    print("-" * 84)
    print(f"{'ИТОГО позитивных':<28} {'':<12} {'':<12} {'':>7} {'':>4} {total:>7.1f} ({total / 60:.1f} мин)")

    print("\nНегативные фикстуры:")
    for path in sorted(NEGATIVE.glob("*")):
        print(f"  {path.name:<24} {path.stat().st_size:>10} байт")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="перегенерировать существующие файлы")
    parser.add_argument("--only", help="сгенерировать только один сценарий по имени файла")
    args = parser.parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            print(f"Не найден {tool} — установите ffmpeg", file=sys.stderr)
            return 1

    TEST_DATA.mkdir(parents=True, exist_ok=True)

    for scenario in SCENARIOS:
        if args.only and scenario.filename != args.only:
            continue
        target = TEST_DATA / scenario.filename
        if target.exists() and not args.force:
            print(f"skip   {scenario.filename} (уже существует, --force для перегенерации)")
            continue
        print(f"build  {scenario.filename} ({len(scenario.lines)} реплик)...")
        with tempfile.TemporaryDirectory() as tmp:
            build_scenario(scenario, Path(tmp))

    telephone = TEST_DATA / TELEPHONE_FILENAME
    if (not telephone.exists() or args.force) and (TEST_DATA / TELEPHONE_SOURCE).exists():
        print(f"build  {TELEPHONE_FILENAME} (телефонный кодек из {TELEPHONE_SOURCE})...")
        build_telephone_copy()

    if not args.only:
        print("build  негативные фикстуры...")
        build_negative_fixtures()

    print_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
