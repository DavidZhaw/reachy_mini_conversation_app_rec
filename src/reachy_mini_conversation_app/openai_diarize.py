import json
import base64
import logging
import mimetypes
from typing import Any
from pathlib import Path
from datetime import datetime, timezone

from openai import AsyncOpenAI


logger = logging.getLogger(__name__)

_AUDIO_REFERENCE_EXTENSIONS = (".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm")


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
    known_speaker_names: list[str] | None = None,
    speaker_references_dir: Path | None = None,
) -> None:
    """Send a WAV file to OpenAI diarizing transcription and save the JSON response."""
    client = AsyncOpenAI(api_key=api_key)
    speaker_names = _normalize_speaker_names(known_speaker_names)
    reference_files = _find_speaker_reference_files(speaker_names, speaker_references_dir)
    extra_body: dict[str, Any] = {}
    if speaker_names:
        extra_body["known_speaker_names"] = speaker_names
    if reference_files:
        extra_body["known_speaker_references"] = [_audio_file_to_data_url(path) for path in reference_files]

    request_kwargs: dict[str, Any] = {
        "file": None,
        "model": model_name,
        "response_format": "diarized_json",
        "chunking_strategy": "auto",
    }
    if extra_body:
        request_kwargs["extra_body"] = extra_body

    with wav_path.open("rb") as audio_file:
        request_kwargs["file"] = audio_file
        response = await client.audio.transcriptions.create(**request_kwargs)

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "provider": "openai",
        "endpoint": "v1/audio/transcriptions",
        "model": model_name,
        "audio_file_name": wav_path.name,
        "known_speaker_names": speaker_names,
        "known_speaker_reference_file_names": [path.name for path in reference_files],
        "response": _serialize_response(response),
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info("Saved audio diarization: %s", output_path)


def _normalize_speaker_names(speaker_names: list[str] | None) -> list[str]:
    """Return non-empty speaker names without duplicates."""
    normalized: list[str] = []
    seen: set[str] = set()
    for name in speaker_names or []:
        stripped = str(name).strip()
        if not stripped or stripped in seen:
            continue
        normalized.append(stripped)
        seen.add(stripped)
    return normalized


def _find_speaker_reference_files(speaker_names: list[str], references_dir: Path | None) -> list[Path]:
    """Find existing reference audio files whose stems match known speaker names."""
    if not speaker_names or references_dir is None or not references_dir.is_dir():
        return []

    desired_stems = {name.lower() for name in speaker_names}
    matches_by_stem: dict[str, Path] = {}
    for path in references_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in _AUDIO_REFERENCE_EXTENSIONS:
            continue
        stem = path.stem.lower()
        if stem in desired_stems and stem not in matches_by_stem:
            matches_by_stem[stem] = path

    return [matches_by_stem[name.lower()] for name in speaker_names if name.lower() in matches_by_stem]


def _audio_file_to_data_url(path: Path) -> str:
    """Return an audio file as a data URL accepted by OpenAI speaker references."""
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type is None:
        mime_type = "audio/wav" if path.suffix.lower() == ".wav" else "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"
