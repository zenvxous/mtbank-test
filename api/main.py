import os
import tempfile
import time
from contextlib import asynccontextmanager

import av
import av.error
import structlog
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from agents.graph import run_analysis
from api.config import settings
from api.logging_config import configure_logging
from api.logging_middleware import AccessLogMiddleware
from asr.diarizer import AudioDiarizer
from asr.transcriber import AudioTranscriber

configure_logging()
log = structlog.get_logger(__name__)

ALLOWED_EXTENSIONS = (".wav", ".mp3", ".ogg")
CHUNK_SIZE = 1024 * 1024

TARGET_SAMPLE_RATE = 16000
TARGET_LAYOUT = "mono"
TARGET_FORMAT = "s16"


class TranscriptSegment(BaseModel):
    speaker: str
    start: float
    end: float
    text: str


class Classification(BaseModel):
    topic: str
    priority: str


class QualityChecklist(BaseModel):
    greeting: bool
    need_detection: bool
    solution_provided: bool
    farewell: bool


class QualityScore(BaseModel):
    total: int
    checklist: QualityChecklist
    comment: str


class ComplianceIssue(BaseModel):
    rule: str
    severity: str
    quote: str
    comment: str


class Compliance(BaseModel):
    passed: bool
    issues: list[ComplianceIssue]


class AnalyzeResponse(BaseModel):
    transcript: list[TranscriptSegment]
    classification: Classification
    quality_score: QualityScore
    compliance: Compliance
    summary: str
    action_items: list[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("app_startup_started", app_name=settings.APP_NAME, app_env=settings.APP_ENV)

    try:
        app.state.transcriber = AudioTranscriber()
    except Exception:
        log.error(
            "transcriber_initialization_failed",
            model_size=settings.WHISPER_MODEL_SIZE,
            exc_info=True,
        )
        raise

    log.debug(
        "transcriber_initialized",
        model_size=settings.WHISPER_MODEL_SIZE,
        device=settings.WHISPER_DEVICE,
        compute_type=settings.WHISPER_COMPUTE_TYPE,
    )

    app.state.diarizer = None

    if settings.DIARIZATION_ENABLED:
        try:
            app.state.diarizer = AudioDiarizer()
            log.debug("diarizer_initialized", model=settings.DIARIZATION_MODEL)
        except Exception:
            log.warning(
                "diarizer_initialization_failed",
                model=settings.DIARIZATION_MODEL,
                exc_info=True,
            )
    else:
        log.debug("diarizer_disabled")

    log.info(
        "app_startup_completed",
        app_name=settings.APP_NAME,
        diarization_loaded=app.state.diarizer is not None,
    )

    yield

    log.info("app_shutdown_started")

    app.state.transcriber = None
    app.state.diarizer = None

    log.info("app_shutdown_completed")


app = FastAPI(
    title="MTBank AI API",
    description="API for AI call analyze",
    lifespan=lifespan,
)

app.add_middleware(AccessLogMiddleware)


@app.get("/health")
def health(request: Request) -> dict:
    return {
        "status": "ok",
        "asr_loaded": request.app.state.transcriber is not None,
        "diarization_loaded": request.app.state.diarizer is not None,
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_audio(request: Request, file: UploadFile = File(...)):
    if not file.filename:
        log.warning("analyze_rejected", reason="no_filename")
        raise HTTPException(status_code=400, detail="No file uploaded")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        log.warning("analyze_rejected", reason="unsupported_extension", filename=file.filename, ext=ext)
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed types are: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    log.info("analyze_started", filename=file.filename, ext=ext)
    started = time.perf_counter()

    temp_file_path, size = _save_upload(file, ext)
    log.debug("upload_saved", filename=file.filename, size_bytes=size)

    try:
        wav_path = _format_audio(temp_file_path)
    finally:
        os.unlink(temp_file_path)

    log.debug("audio_converted", sample_rate=TARGET_SAMPLE_RATE, layout=TARGET_LAYOUT)

    try:
        transcribe_started = time.perf_counter()
        transcript = request.app.state.transcriber.transcribe(wav_path)
        log.info(
            "transcription_completed",
            segments=len(transcript.segments),
            language=transcript.language,
            audio_duration_s=transcript.duration,
            duration_ms=round((time.perf_counter() - transcribe_started) * 1000, 2),
        )

        diarizer = request.app.state.diarizer
        if diarizer is not None:
            try:
                diarize_started = time.perf_counter()
                diarizer.assign_speakers(transcript.segments, wav_path)
                log.info(
                    "diarization_completed",
                    segments=len(transcript.segments),
                    duration_ms=round((time.perf_counter() - diarize_started) * 1000, 2),
                )
            except Exception:
                log.warning("diarization_failed", filename=file.filename, exc_info=True)
        else:
            log.debug("diarization_skipped", filename=file.filename)
    finally:
        os.unlink(wav_path)

    dialog = [segment.as_dict() for segment in transcript.segments]

    final_analysis = run_analysis(dialog)

    log.info(
        "analyze_completed",
        filename=file.filename,
        segments=len(dialog),
        duration_ms=round((time.perf_counter() - started) * 1000, 2),
    )

    return final_analysis

def _format_audio(file_path: str) -> str:
    out_fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(out_fd)

    try:
        with av.open(file_path) as src:
            if not src.streams.audio:
                log.warning("audio_conversion_failed", reason="no_audio_stream")
                raise HTTPException(status_code=422, detail="File contains no audio stream")

            resampler = av.AudioResampler(
                format=TARGET_FORMAT, layout=TARGET_LAYOUT, rate=TARGET_SAMPLE_RATE
            )

            with av.open(out_path, mode="w", format="wav") as dst:
                out_stream = dst.add_stream(
                    "pcm_s16le", rate=TARGET_SAMPLE_RATE, layout=TARGET_LAYOUT
                )

                def _mux(frames):
                    for frame in frames:
                        frame.pts = None
                        dst.mux(out_stream.encode(frame))

                for frame in src.decode(src.streams.audio[0]):
                    _mux(resampler.resample(frame))

                _mux(resampler.resample(None))
                dst.mux(out_stream.encode(None))
    except HTTPException:
        os.unlink(out_path)
        raise
    except av.error.FFmpegError as e:
        os.unlink(out_path)
        log.warning("audio_conversion_failed", reason="decode_error", error=str(e))
        raise HTTPException(status_code=422, detail=f"Failed to decode audio: {e}") from e
    except BaseException:
        os.unlink(out_path)
        raise

    return out_path


def _save_upload(file: UploadFile, ext: str) -> tuple[str, int]:
    size = 0
    file.file.seek(0)

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        temp_file_path = tmp.name
        try:
            while chunk := file.file.read(CHUNK_SIZE):
                size += len(chunk)
                if size > settings.MAX_UPLOAD_BYTES:
                    log.warning(
                        "upload_rejected",
                        reason="too_large",
                        filename=file.filename,
                        limit_bytes=settings.MAX_UPLOAD_BYTES,
                    )
                    raise HTTPException(
                        status_code=413,
                        detail=f"File is too large, limit is {settings.MAX_UPLOAD_BYTES} bytes",
                    )
                tmp.write(chunk)
        except BaseException:
            os.unlink(temp_file_path)
            raise

    if size == 0:
        os.unlink(temp_file_path)
        log.warning("upload_rejected", reason="empty_file", filename=file.filename)
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    return temp_file_path, size
