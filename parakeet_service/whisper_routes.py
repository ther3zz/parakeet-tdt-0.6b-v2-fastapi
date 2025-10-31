import asyncio
import tempfile
from pathlib import Path
from collections import defaultdict
from typing import Optional

from fastapi import (
    APIRouter, BackgroundTasks, HTTPException, Request, status,
    File, UploadFile, Form
)

from .audio import schedule_cleanup
from .model import _to_builtin
from .schemas import WhisperTranscriptionResponse
from .config import logger, BATCH_SIZE, TARGET_SR
from .chunker import vad_chunk_lowmem


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
    output: Optional[str] = Form("json"),
):
    """
    Trancribes an audio stream. It intelligently detects if the stream is a
    complete WAV file or raw PCM data, and processes it accordingly.
    """
    source_description = f"uploaded file '{audio_file.filename}'"

    # Read the entire raw audio stream into memory.
    content = await audio_file.read()
    total_bytes_read = len(content)
    await audio_file.close()

    logger.info(f"Successfully read {total_bytes_read} bytes from uploaded file into memory.")
    if total_bytes_read == 0:
        logger.error("File upload failed: received file was empty.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded 'audio_file' is empty.")

    # A valid WAV file starts with "RIFF" and has "WAVE" at offset 8.
    is_valid_wav = (
        total_bytes_read > 12 and
        content[:4] == b'RIFF' and
        content[8:12] == b'WAVE'
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav_file:
        tmp_path = Path(tmp_wav_file.name)

        if is_valid_wav:
            logger.info("Valid WAV header detected. Writing file directly, bypassing FFmpeg.")
            tmp_wav_file.write(content)
        else:
            logger.info("No WAV header detected. Assuming raw PCM stream and using FFmpeg to create a valid WAV file.")
            # This FFmpeg command reads raw PCM data from stdin and correctly wraps it into a valid WAV file.
            ffmpeg_cmd = [
                "ffmpeg", "-v", "error", "-nostdin", "-y",
                "-f", "s16le",             # Format: signed 16-bit little-endian
                "-ar", str(TARGET_SR),     # Sample Rate
                "-ac", "1",                # Audio Channels: 1 (Mono)
                "-i", "-",                 # Input: stdin
                "-acodec", "pcm_s16le",    # Output codec
                "-f", "wav", str(tmp_path) # Output format and path
            ]
            logger.debug(f"Executing FFmpeg command: {' '.join(ffmpeg_cmd)}")

            process = await asyncio.create_subprocess_exec(
                *ffmpeg_cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await process.communicate(input=content)

            if process.returncode != 0:
                stderr_str = stderr.decode().strip()
                logger.error(f"FFmpeg failed while wrapping RAW audio for {source_description} with return code {process.returncode}")
                logger.error(f"FFmpeg error output: {stderr_str}")
                raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=f"FFmpeg processing failed: {stderr_str[:250]}")

    # At this point, tmp_path always points to a valid WAV file.
    # Chunk the audio file using VAD
    chunk_paths = vad_chunk_lowmem(tmp_path) or [tmp_path]
    logger.info(f"Processing success. Sending {len(chunk_paths)} chunks to ASR from {source_description}")

    # Schedule all temporary files for cleanup
    cleanup_files = [tmp_path] + [p for p in chunk_paths if p != tmp_path]
    schedule_cleanup(background_tasks, *cleanup_files)

    # Run the ASR model
    model = request.app.state.asr_model
    try:
        outs = model.transcribe(
            [str(p) for p in chunk_paths],
            batch_size=BATCH_SIZE,
            timestamps=True,
        )
    except RuntimeError as exc:
        logger.exception("ASR transcription failed")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    # Process and aggregate results
    if isinstance(outs, tuple):
        outs = outs[0]

    texts, merged_timestamps = [], defaultdict(list)
    for h in outs:
        texts.append(getattr(h, "text", str(h)))
        for k, v in _to_builtin(getattr(h, "timestamp", {})).items():
            merged_timestamps[k].extend(v)

    # Format the response
    return WhisperTranscriptionResponse(
        text=" ".join(texts).strip(),
        language="en",
        segments=merged_timestamps.get("segment", []),
    )
