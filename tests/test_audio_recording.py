import json
import wave
import asyncio
from typing import Any
from pathlib import Path

import numpy as np
import pytest

from reachy_mini_conversation_app.gemini_live import GeminiLiveHandler
from reachy_mini_conversation_app.audio_recording import ConversationAudioRecorder
from reachy_mini_conversation_app.openai_realtime import OpenaiRealtimeHandler
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.huggingface_realtime import HuggingFaceRealtimeHandler


def _read_manifest(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def test_audio_recorder_writes_per_turn_wavs_and_manifest(tmp_path: Path) -> None:
    """Recorder should write turn WAV files and timeline metadata."""
    recorder = ConversationAudioRecorder(tmp_path, sample_rate=24_000)

    recorder.buffer_sent_input_audio(np.array([1, 2, 3, 4], dtype=np.int16))
    recorder.start_user_turn()
    recorder.buffer_sent_input_audio(np.array([5, 6], dtype=np.int16))
    recorder.finish_user_turn(transcript="hello")

    recorder.append_assistant_audio(np.array([[10, 11, 12]], dtype=np.int16))
    recorder.finish_assistant_turn()

    manifest = _read_manifest(recorder.run_dir)
    entries = manifest["entries"]

    assert [entry["file_name"] for entry in entries] == [
        "turn_0001_user_input.wav",
        "turn_0001_assistant_output.wav",
    ]
    assert entries[0]["direction"] == "user_input"
    assert entries[0]["transcript"] == "hello"
    assert entries[0]["duration_seconds"] == pytest.approx(6 / 24_000)
    assert entries[0]["started_at"] == entries[0]["recorded_at"]
    assert entries[0]["start_offset_seconds"] >= 0
    assert entries[1]["direction"] == "assistant_output"

    with wave.open(str(recorder.run_dir / "turn_0001_user_input.wav"), "rb") as wav_file:
        assert wav_file.getframerate() == 24_000
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getnframes() == 6


def test_audio_recorder_reports_saved_user_turn_for_diarization(tmp_path: Path) -> None:
    """Recorder should expose saved user WAV paths for asynchronous diarization."""
    saved: list[tuple[Path, dict[str, Any]]] = []
    recorder = ConversationAudioRecorder(
        tmp_path,
        sample_rate=24_000,
        on_user_turn_saved=lambda wav_path, entry: saved.append((wav_path, entry)),
    )

    recorder.start_user_turn()
    recorder.buffer_sent_input_audio(np.array([1, 2], dtype=np.int16))
    recorder.finish_user_turn(transcript="hello")

    assert len(saved) == 1
    wav_path, entry = saved[0]
    assert wav_path.name == "turn_0001_user_input.wav"
    assert entry["diarization_file_name"] == "turn_0001_user_input.diarization.json"
    manifest = _read_manifest(recorder.run_dir)
    assert manifest["entries"][0]["diarization_file_name"] == "turn_0001_user_input.diarization.json"


@pytest.mark.asyncio
async def test_realtime_handler_records_only_audio_sent_successfully(tmp_path: Path, monkeypatch: Any) -> None:
    """Realtime handlers should record input only after the append call succeeds."""
    monkeypatch.setenv("AUDIO_RECORDINGS_DIR", str(tmp_path))

    class FakeInputAudioBuffer:
        def __init__(self) -> None:
            self.appended: list[str] = []

        async def append(self, *, audio: str) -> None:
            self.appended.append(audio)

    class FakeConnection:
        def __init__(self) -> None:
            self.input_audio_buffer = FakeInputAudioBuffer()

    deps = ToolDependencies(reachy_mini=None, movement_manager=None)
    handler = OpenaiRealtimeHandler(deps, record_audio=True)
    handler.connection = FakeConnection()  # type: ignore[assignment]

    await handler.receive((24_000, np.array([100, 200, 300], dtype=np.int16)))
    handler._start_recorded_user_audio_turn()
    handler._finish_recorded_user_audio_turn("sent")

    assert handler.connection.input_audio_buffer.appended  # type: ignore[union-attr]
    assert handler._audio_recorder is not None
    manifest = _read_manifest(handler._audio_recorder.run_dir)
    assert manifest["entries"][0]["file_name"] == "turn_0001_user_input.wav"
    assert manifest["entries"][0]["duration_seconds"] == pytest.approx(3 / 24_000)

    with wave.open(str(handler._audio_recorder.run_dir / "turn_0001_user_input.wav"), "rb") as wav_file:
        assert wav_file.getnframes() == 3


@pytest.mark.asyncio
async def test_realtime_audio_recording_is_disabled_by_default(tmp_path: Path, monkeypatch: Any) -> None:
    """Realtime handlers should not create recording folders unless explicitly enabled."""
    monkeypatch.setenv("AUDIO_RECORDINGS_DIR", str(tmp_path))

    deps = ToolDependencies(reachy_mini=None, movement_manager=None)
    handler = HuggingFaceRealtimeHandler(deps)

    handler._record_sent_input_audio(np.array([100, 200, 300], dtype=np.int16))
    handler._start_recorded_user_audio_turn()
    handler._finish_recorded_user_audio_turn("ignored")
    handler._record_received_assistant_audio(np.array([1, 2], dtype=np.int16))
    handler._finish_recorded_assistant_audio_turn()

    assert handler._audio_recorder is None
    assert list(tmp_path.iterdir()) == []


def test_gemini_audio_recording_preserves_input_and_output_sample_rates(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Gemini recordings should preserve its distinct input and output sample rates."""
    monkeypatch.setenv("AUDIO_RECORDINGS_DIR", str(tmp_path))

    deps = ToolDependencies(reachy_mini=None, movement_manager=None)
    handler = GeminiLiveHandler(deps, record_audio=True)

    handler._record_sent_input_audio(np.array([1, 2, 3], dtype=np.int16))
    handler._finish_recorded_user_audio_turn()
    handler._record_received_assistant_audio(np.array([4, 5, 6, 7], dtype=np.int16))
    handler._finish_recorded_assistant_audio_turn()

    assert handler._audio_recorder is not None
    manifest = _read_manifest(handler._audio_recorder.run_dir)
    assert manifest["entries"][0]["direction"] == "user_input"
    assert manifest["entries"][0]["sample_rate"] == 16_000
    assert manifest["entries"][0]["duration_seconds"] == pytest.approx(3 / 16_000)
    assert manifest["entries"][1]["direction"] == "assistant_output"
    assert manifest["entries"][1]["sample_rate"] == 24_000
    assert manifest["entries"][1]["duration_seconds"] == pytest.approx(4 / 24_000)


@pytest.mark.asyncio
async def test_realtime_audio_diarization_runs_after_user_wav_is_saved(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Diarization should run asynchronously after a user WAV has been written."""
    monkeypatch.setenv("AUDIO_RECORDINGS_DIR", str(tmp_path))
    monkeypatch.setattr("reachy_mini_conversation_app.config.config.OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("reachy_mini_conversation_app.config.config.DIARIZATION_MODEL_NAME", "test-diarize-model")

    calls: list[tuple[Path, Path, str, str]] = []

    async def fake_diarize_audio_file(
        *,
        wav_path: Path,
        output_path: Path,
        model_name: str,
        api_key: str,
    ) -> None:
        calls.append((wav_path, output_path, model_name, api_key))
        output_path.write_text('{"ok": true}\n', encoding="utf-8")

    monkeypatch.setattr("reachy_mini_conversation_app.openai_diarize.diarize_audio_file", fake_diarize_audio_file)

    deps = ToolDependencies(reachy_mini=None, movement_manager=None)
    handler = OpenaiRealtimeHandler(deps, record_audio=True, record_diarize_audio=True)

    handler._start_recorded_user_audio_turn()
    handler._record_sent_input_audio(np.array([1, 2, 3], dtype=np.int16))
    handler._finish_recorded_user_audio_turn("hello")
    await asyncio.sleep(0)
    await handler._wait_for_audio_diarization()

    assert len(calls) == 1
    wav_path, output_path, model_name, api_key = calls[0]
    assert wav_path.name == "turn_0001_user_input.wav"
    assert output_path.name == "turn_0001_user_input.diarization.json"
    assert model_name == "test-diarize-model"
    assert api_key == "test-key"
    assert output_path.exists()
