import logging
import os
import tempfile
from contextlib import asynccontextmanager

import av
import av.error
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from api.config import settings
from asr.diarizer import AudioDiarizer
from asr.transcriber import AudioTranscriber

logger = logging.getLogger(__name__)

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


class AnalyzeResponse(BaseModel):
    transcript: list[TranscriptSegment]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.transcriber = AudioTranscriber()
    app.state.diarizer = None

    if settings.DIARIZATION_ENABLED:
        try:
            app.state.diarizer = AudioDiarizer()
        except Exception as e:
            logger.warning("Диаризация отключена, пайплайн не загрузился: %s", e)

    yield


app = FastAPI(
    title="MTBank AI API",
    description="API for AI call analyze",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request) -> dict:
    return {
        "status": "ok",
        "asr_loaded": request.app.state.transcriber is not None,
        "diarization_loaded": request.app.state.diarizer is not None,
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_audio(request: Request, file: UploadFile = File(...)) -> AnalyzeResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed types are: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    temp_file_path, size = _save_upload(file, ext)

    try:
        wav_path = _format_audio(temp_file_path)
    finally:
        os.unlink(temp_file_path)

    try:
        transcript = request.app.state.transcriber.transcribe(wav_path)

        diarizer = request.app.state.diarizer
        if diarizer is not None:
            try:
                diarizer.assign_speakers(transcript.segments, wav_path)
            except Exception as e:
                logger.warning("Диаризация не удалась для %s: %s", file.filename, e)
    finally:
        os.unlink(wav_path)

    return AnalyzeResponse(
        transcript=[segment.as_dict() for segment in transcript.segments],
    )

def _format_audio(file_path: str) -> str:
    out_fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(out_fd)

    try:
        with av.open(file_path) as src:
            if not src.streams.audio:
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
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    return temp_file_path, size
