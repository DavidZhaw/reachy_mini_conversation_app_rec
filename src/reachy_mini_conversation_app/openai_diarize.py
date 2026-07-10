import json
import logging
from typing import Any
from pathlib import Path
from datetime import datetime, timezone

from openai import AsyncOpenAI


logger = logging.getLogger(__name__)


def _serialize_response(response: Any) -> Any:
    if hasattr(response, "model_dump") and callable(response.model_dump):
        return response.model_dump(mode="json")
    if hasattr(response, "to_dict") and callable(response.to_dict):
        return response.to_dict()
    if isinstance(response, str):
        return {"text": response}
    try:
        return dict(response)
    except Exception:
        return {"raw": repr(response)}


async def diarize_audio_file(
    *,
    wav_path: Path,
    output_path: Path,
    model_name: str,
    api_key: str,
) -> None:
    """Send a WAV file to OpenAI diarizing transcription and save the JSON response."""
    client = AsyncOpenAI(api_key=api_key)
    with wav_path.open("rb") as audio_file:
        response = await client.audio.transcriptions.create(
            file=audio_file,
            model=model_name,
            response_format="json",
        )

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "provider": "openai",
        "endpoint": "v1/audio/transcriptions",
        "model": model_name,
        "audio_file_name": wav_path.name,
        "response": _serialize_response(response),
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info("Saved audio diarization: %s", output_path)
