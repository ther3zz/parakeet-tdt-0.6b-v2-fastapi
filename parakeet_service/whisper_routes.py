import asyncio
import tempfile
from pathlib import Path
from collections import defaultdict

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request, status

from .audio import schedule_cleanup
from .model import _to_builtin
from .schemas import WhisperTranscriptionResponse
from .config import logger, BATCH_SIZE
from .chunker import vad_chunk_lowmem


whisper_router = APIRouter(tags=["whisper_compatibility"])


@whisper_router.post(
    "/audio/transcriptions",
    response_model=WhisperTranscriptionResponse,
    summary="Transcribe a local media file (Whisper ASR compatible)",
)
@whisper_router.post(
    "/audio/transcriptions/detect-language",
    response_model=WhisperTranscriptionResponse,
    summary="Transcribe a local media file (Whisper ASR compatible)",
)
async def transcribe_local_file(
    request: Request,
    background_tasks: BackgroundTasks,
    video_file: str = Query(..., alias="video_file", description="Path to the local video/audio file to transcribe."),
    encode: bool = Query(False, description="Not used. For compatibility."),
):
    """
    Trancribes a local audio or video file and returns the transcription in a
    Whisper ASR compatible format.

    This endpoint uses ffmpeg to extract and convert audio to a processable format.
    """
    local_file_path = Path(video_file)
    if not local_file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found at path: {video_file}",
        )

    # Create a temporary file to store the converted WAV audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp_path = Path(tmp.name)

    # Use FFmpeg to convert the source file to 16kHz mono WAV
    ffmpeg_cmd = [
        "ffmpeg", "-v", "error", "-nostdin", "-y",
        "-i", str(local_file_path),
        "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
        "-f", "wav", str(tmp_path)
    ]

    try:
        process = await asyncio.create_subprocess_exec(
            *ffmpeg_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await process.communicate()

        if process.returncode != 0:
            stderr_str = stderr.decode().strip()
            logger.error(f"FFmpeg failed with return code {process.returncode} for file '{video_file}'")
            logger.error(f"FFmpeg error output: {stderr_str}")
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported media file: {stderr_str[:250]}",
            )
    except Exception as e:
        logger.exception("An error occurred during FFmpeg processing.")
        if tmp_path.exists():
            tmp_path.unlink() # Ensure cleanup on error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio processing failed: {str(e)}",
        )

    # Chunk the converted audio file using VAD for better processing of long audio
    chunk_paths = vad_chunk_lowmem(tmp_path) or [tmp_path]
    logger.info("transcribe_local_file(): sending %d chunks to ASR", len(chunk_paths))

    # Schedule all temporary files for cleanup
    cleanup_files = [tmp_path] + [p for p in chunk_paths if p != tmp_path]
    schedule_cleanup(background_tasks, *cleanup_files)

    # Run the ASR model on the audio chunks
    model = request.app.state.asr_model
    try:
        # Timestamps are enabled to populate the 'segments' field in the response
        outs = model.transcribe(
            [str(p) for p in chunk_paths],
            batch_size=BATCH_SIZE,
            timestamps=True,
        )
    except RuntimeError as exc:
        logger.exception("ASR transcription failed")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc)
        ) from exc

    # Process and aggregate results from all chunks
    if isinstance(outs, tuple):
        outs = outs[0]

    texts = []
    merged_timestamps = defaultdict(list)
    for h in outs:
        texts.append(getattr(h, "text", str(h)))
        # Note: Timestamps from chunks are concatenated. For accurate absolute timing
        # in long files, a more advanced chunking and timestamp recombination
        # strategy would be required. This matches the existing service's behavior.
        for k, v in _to_builtin(getattr(h, "timestamp", {})).items():
            merged_timestamps[k].extend(v)

    merged_text = " ".join(texts).strip()

    # Format the response to be compatible with the whisper-asr-webservice output
    return WhisperTranscriptionResponse(
        text=merged_text,
        language="en",  # Parakeet is an English-only model
        segments=merged_timestamps.get("segment", [])
    )
