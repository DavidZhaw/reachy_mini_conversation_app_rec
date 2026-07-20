import json
from typing import Any
from pathlib import Path

import pytest

import reachy_mini_conversation_app.openai_diarize as diarize_mod


@pytest.mark.asyncio
async def test_diarize_audio_file_saves_openai_transcription_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI diarization wrapper should save request metadata and SDK response JSON."""
    wav_path = tmp_path / "turn_0001_user_input.wav"
    wav_path.write_bytes(b"RIFFfake")
    output_path = tmp_path / "turn_0001_user_input.diarization.json"
    calls: list[dict[str, Any]] = []

    class FakeResponse:
        def model_dump(self, *, mode: str) -> dict[str, Any]:
            return {"mode": mode, "text": "hello", "segments": [{"speaker": "speaker_0", "text": "hello"}]}

    class FakeTranscriptions:
        async def create(self, **kwargs: Any) -> FakeResponse:
            calls.append(kwargs)
            return FakeResponse()

    class FakeClient:
        def __init__(self, *, api_key: str) -> None:
            self.api_key = api_key
            self.audio = type("Audio", (), {"transcriptions": FakeTranscriptions()})()

    monkeypatch.setattr(diarize_mod, "AsyncOpenAI", FakeClient)

    await diarize_mod.diarize_audio_file(
        wav_path=wav_path,
        output_path=output_path,
        model_name="test-diarize-model",
        api_key="test-key",
    )

    assert calls[0]["model"] == "test-diarize-model"
    assert calls[0]["response_format"] == "diarized_json"
    assert calls[0]["chunking_strategy"] == "auto"
    assert "extra_body" not in calls[0]
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["provider"] == "openai"
    assert payload["endpoint"] == "v1/audio/transcriptions"
    assert payload["model"] == "test-diarize-model"
    assert payload["audio_file_name"] == "turn_0001_user_input.wav"
    assert payload["known_speaker_names"] == []
    assert payload["known_speaker_reference_file_names"] == []
    assert payload["response"]["segments"][0]["speaker"] == "speaker_0"


@pytest.mark.asyncio
async def test_diarize_audio_file_sends_known_speakers_and_available_references(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Known speaker names and matching reference clips should be sent to OpenAI."""
    wav_path = tmp_path / "turn_0001_user_input.wav"
    wav_path.write_bytes(b"RIFFfake")
    references_dir = tmp_path / "speaker_references"
    references_dir.mkdir()
    (references_dir / "bob.wav").write_bytes(b"bob audio")
    output_path = tmp_path / "turn_0001_user_input.diarization.json"
    calls: list[dict[str, Any]] = []

    class FakeTranscriptions:
        async def create(self, **kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs)
            return {"segments": []}

    class FakeClient:
        def __init__(self, *, api_key: str) -> None:
            self.audio = type("Audio", (), {"transcriptions": FakeTranscriptions()})()

    monkeypatch.setattr(diarize_mod, "AsyncOpenAI", FakeClient)

    await diarize_mod.diarize_audio_file(
        wav_path=wav_path,
        output_path=output_path,
        model_name="test-diarize-model",
        api_key="test-key",
        known_speaker_names=["bob", "alice"],
        speaker_references_dir=references_dir,
    )

    extra_body = calls[0]["extra_body"]
    assert extra_body["known_speaker_names"] == ["bob", "alice"]
    assert extra_body["known_speaker_references"][0].startswith("data:audio/")
    assert extra_body["known_speaker_references"][0].endswith(";base64,Ym9iIGF1ZGlv")

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["known_speaker_names"] == ["bob", "alice"]
    assert payload["known_speaker_reference_file_names"] == ["bob.wav"]
