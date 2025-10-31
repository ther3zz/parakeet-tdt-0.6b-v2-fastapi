import asyncio
import tempfile
from pathlib import Path
from collections import defaultdict
from typing import Optional
from io import StringIO

from fastapi import (
    APIRouter, BackgroundTasks, HTTPException, Request, status,
    File, UploadFile, Form, Response
)
from fastapi.responses import PlainTextResponse, JSONResponse

from .audio import schedule_cleanup
from .model import _to_builtin
from .schemas import WhisperTranscriptionResponse
from .config import logger, BATCH_SIZE, TARGET_SR
from .chunker import vad_chunk_lowmem
from .formatters import write_txt, write_vtt, write_srt, write_json


whisper_router = APIRouter(tags=["whisper_compatibility"])


@whisper_router.post(
    "/asr",
    response_model=WhisperTranscriptionResponse,
    summary="Transcribe an audio or video file (For use with Bazarr)",
)
@whisper_router.post(
    "/audio/transcriptions",
    response_model=WhisperTranscriptionResponse,
    summary="[OpenAI Compatible] Transcribe audio",
)
@whisper_router.post(
    "/audio/transcriptions/detect-language",
    response_model=WhisperTranscriptionResponse,
    summary="[Legacy Alias] Transcribe audio",
    deprecated=True,
)
async def transcribe_asr_compatible(
    request: Request,
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(..., description="Audio or video file to transcribe."),
    task: Optional[str] = Form("transcribe"),
    language: Optional[str] = Form("en"),
    # The 'output' parameter is now crucial for determining the response format.
    output: str = Form("json", enum=["txt", "vtt", "srt", "json"]),
) -> Response: # Return type is now a generic Response
    """
    Trancribes an audio stream and returns the result in the format specified
    by the 'output' parameter (json, srt, vtt, or txt).
    """
    source_description = f"uploaded file '{audio_file.filename}'"

    content = await audio_file.read()
    total_bytes_read = len(content)
    await audio_file.close()

    if total_bytes_read == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded 'audio_file' is empty.")

    is_valid_wav = (total_bytes_read > 12 and content[:4] == b'RIFF' and content[8:12] == b'WAVE')

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav_file:
        tmp_path = Path(tmp_wav_file.name)
        if is_valid_wav:
            tmp_wav_file.write(content)
        else:
            ffmpeg_cmd = [
                "ffmpeg", "-v", "error", "-nostdin", "-y",
                "-f", "s16le", "-ar", str(TARGET_SR), "-ac", "1",
                "-i", "-",
                "-acodec", "pcm_s16le", "-f", "wav", str(tmp_path)
            ]
            process = await asyncio.create_subprocess_exec(
                *ffmpeg_cmd, stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await process.communicate(input=content)

            if process.returncode != 0:
                stderr_str = stderr.decode().strip()
                raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=f"FFmpeg processing failed: {stderr_str[:250]}")

    chunk_paths = vad_chunk_lowmem(tmp_path) or [tmp_path]
    cleanup_files = [tmp_path] + [p for p in chunk_paths if p != tmp_path]
    schedule_cleanup(background_tasks, *cleanup_files)

    model = request.app.state.asr_model
    try:
        outs = model.transcribe([str(p) for p in chunk_paths], batch_size=BATCH_SIZE, timestamps=True)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    if isinstance(outs, tuple):
        outs = outs[0]

    texts, merged_timestamps = [], defaultdict(list)
    for h in outs:
        texts.append(getattr(h, "text", str(h)))
        for k, v in _to_builtin(getattr(h, "timestamp", {})).items():
            merged_timestamps[k].extend(v)

    
    # Create the final result dictionary in the expected structure.
    final_result = {
        "text": " ".join(texts).strip(),
        "segments": merged_timestamps.get("segment", []),
        "language": "en"
    }

    # Use the appropriate formatter based on the 'output' parameter.
    with StringIO() as string_io:
        if output == "srt":
            write_srt(final_result, string_io)
            return PlainTextResponse(string_io.getvalue(), media_type="text/plain")
        elif output == "vtt":
            write_vtt(final_result, string_io)
            return PlainTextResponse(string_io.getvalue(), media_type="text/plain")
        elif output == "txt":
            write_txt(final_result, string_io)
            return PlainTextResponse(string_io.getvalue(), media_type="text/plain")
        else: # Default to JSON
            # We still use our Pydantic model here for validation before returning a JSONResponse
            validated_result = WhisperTranscriptionResponse(**final_result)
            return JSONResponse(validated_result.model_dump())
    
