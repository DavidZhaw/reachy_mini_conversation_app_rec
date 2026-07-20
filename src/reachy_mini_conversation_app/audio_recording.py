import json
import wave
import logging
from typing import Any, Callable
from pathlib import Path
from datetime import datetime, timezone
from collections import deque
from dataclasses import field, dataclass

import numpy as np
from numpy.typing import NDArray


logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_run_dir_name(dt: datetime) -> str:
    return dt.astimezone().strftime("%Y%m%d_%H%M%S_%f")


def _isoformat(dt: datetime) -> str:
    return dt.isoformat(timespec="microseconds")


@dataclass
class _AudioChunk:
    data: bytes
    sample_count: int
    recorded_at: datetime


@dataclass
class _AudioTurn:
    direction: str
    turn_index: int
    file_name: str
    started_at: datetime
    sample_rate: int
    chunks: list[_AudioChunk] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def sample_count(self) -> int:
        return sum(chunk.sample_count for chunk in self.chunks)

    @property
    def ended_at(self) -> datetime:
        return self.chunks[-1].recorded_at if self.chunks else self.started_at


class ConversationAudioRecorder:
    """Persist realtime conversation audio as per-turn WAV files."""

    def __init__(
        self,
        root_dir: Path,
        *,
        sample_rate: int,
        channels: int = 1,
        sample_width_bytes: int = 2,
        input_preroll_seconds: float = 1.0,
        on_user_turn_saved: Callable[[Path, dict[str, Any]], None] | None = None,
    ) -> None:
        """Create a per-run recording directory and manifest."""
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width_bytes = sample_width_bytes
        self.input_preroll_samples = int(sample_rate * input_preroll_seconds)
        self.on_user_turn_saved = on_user_turn_saved

        self.started_at = _utc_now()
        self.run_dir = self.root_dir / _format_run_dir_name(self.started_at)
        self.manifest_path = self.run_dir / "manifest.json"
        self.entries: list[dict[str, Any]] = []
        self._user_turn_index = 0
        self._assistant_turn_index = 0
        self._current_user_turn: _AudioTurn | None = None
        self._current_assistant_turn: _AudioTurn | None = None
        self._input_preroll: deque[_AudioChunk] = deque()
        self._input_preroll_sample_count = 0

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._write_manifest()

    def buffer_sent_input_audio(self, audio_frame: NDArray[np.int16], *, sample_rate: int | None = None) -> None:
        """Record input audio that was successfully sent to the realtime backend."""
        chunk = self._chunk_from_audio_frame(audio_frame)
        if chunk.sample_count == 0:
            return

        if self._current_user_turn is not None:
            self._current_user_turn.chunks.append(chunk)
        else:
            self._input_preroll.append(chunk)
            self._input_preroll_sample_count += chunk.sample_count
            while self._input_preroll_sample_count > self.input_preroll_samples and self._input_preroll:
                removed = self._input_preroll.popleft()
                self._input_preroll_sample_count -= removed.sample_count

    def start_user_turn(self, *, sample_rate: int | None = None) -> None:
        """Start a user input turn, including recent audio already sent before VAD fired."""
        if self._current_user_turn is not None:
            return

        self._user_turn_index += 1
        chunks = list(self._input_preroll)
        self._input_preroll.clear()
        self._input_preroll_sample_count = 0

        started_at = chunks[0].recorded_at if chunks else _utc_now()
        file_name = f"turn_{self._user_turn_index:04d}_user_input.wav"
        self._current_user_turn = _AudioTurn(
            direction="user_input",
            turn_index=self._user_turn_index,
            file_name=file_name,
            started_at=started_at,
            sample_rate=sample_rate or self.sample_rate,
            chunks=chunks,
        )

    def finish_user_turn(self, *, transcript: str | None = None) -> None:
        """Persist the active user turn, if one exists."""
        turn = self._current_user_turn
        self._current_user_turn = None
        if turn is None:
            return
        if transcript is not None:
            turn.metadata["transcript"] = transcript
        self._finish_turn(turn)

    def append_assistant_audio(self, audio_frame: NDArray[np.int16], *, sample_rate: int | None = None) -> None:
        """Record assistant output audio received from the realtime backend."""
        chunk = self._chunk_from_audio_frame(audio_frame)
        if chunk.sample_count == 0:
            return

        if self._current_assistant_turn is None:
            self._assistant_turn_index += 1
            file_name = f"turn_{self._assistant_turn_index:04d}_assistant_output.wav"
            self._current_assistant_turn = _AudioTurn(
                direction="assistant_output",
                turn_index=self._assistant_turn_index,
                file_name=file_name,
                started_at=chunk.recorded_at,
                sample_rate=sample_rate or self.sample_rate,
            )
        self._current_assistant_turn.chunks.append(chunk)

    def finish_assistant_turn(self) -> None:
        """Persist the active assistant turn, if one exists."""
        turn = self._current_assistant_turn
        self._current_assistant_turn = None
        if turn is not None:
            self._finish_turn(turn)

    def close(self) -> None:
        """Flush any in-progress turns and rewrite the manifest."""
        self.finish_user_turn()
        self.finish_assistant_turn()
        self._write_manifest()

    def _finish_turn(self, turn: _AudioTurn) -> None:
        if turn.sample_count == 0:
            logger.debug("Skipping empty audio recording turn: %s", turn.file_name)
            self._write_manifest()
            return

        path = self.run_dir / turn.file_name
        try:
            with wave.open(str(path), "wb") as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(self.sample_width_bytes)
                wav_file.setframerate(turn.sample_rate)
                for chunk in turn.chunks:
                    wav_file.writeframes(chunk.data)
        except Exception as exc:
            logger.warning("Failed to write audio recording %s: %s", path, exc)
            return

        duration_seconds = turn.sample_count / turn.sample_rate
        entry = {
            "direction": turn.direction,
            "turn_index": turn.turn_index,
            "file_name": turn.file_name,
            "recorded_at": _isoformat(turn.started_at),
            "started_at": _isoformat(turn.started_at),
            "ended_at": _isoformat(turn.ended_at),
            "start_offset_seconds": (turn.started_at - self.started_at).total_seconds(),
            "duration_seconds": duration_seconds,
            "sample_rate": turn.sample_rate,
            "channels": self.channels,
            "sample_width_bytes": self.sample_width_bytes,
            "sample_format": "pcm_s16le",
            **turn.metadata,
        }
        if turn.direction == "user_input" and self.on_user_turn_saved is not None:
            entry["diarization_file_name"] = path.with_suffix(".diarization.json").name
        if turn.direction == "user_input":
            entry["speaker_name"] = "unknown"
        self.entries.append(entry)
        self._write_manifest()
        if turn.direction == "user_input" and self.on_user_turn_saved is not None:
            self.on_user_turn_saved(path, entry)
        logger.info("Saved realtime audio recording: %s", path)

    def update_user_turn_speaker_name(self, *, file_name: str, speaker_name: str) -> bool:
        """Update one saved user-input manifest entry with a diarized speaker name."""
        normalized_speaker_name = (speaker_name or "").strip() or "unknown"
        for entry in self.entries:
            if entry.get("direction") != "user_input" or entry.get("file_name") != file_name:
                continue
            entry["speaker_name"] = normalized_speaker_name
            self._write_manifest()
            return True
        return False

    def _write_manifest(self) -> None:
        manifest = {
            "schema_version": 1,
            "run_started_at": _isoformat(self.started_at),
            "recording_directory": str(self.run_dir),
            "audio_format": {
                "container": "wav",
                "encoding": "pcm_s16le",
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "sample_width_bytes": self.sample_width_bytes,
            },
            "entries": self.entries,
        }
        try:
            self.manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to write audio recording manifest %s: %s", self.manifest_path, exc)

    def _chunk_from_audio_frame(self, audio_frame: NDArray[np.int16]) -> _AudioChunk:
        audio = np.asarray(audio_frame, dtype=np.int16)
        if audio.ndim == 2:
            if audio.shape[0] == 1:
                audio = audio.reshape(-1)
            elif audio.shape[1] == 1:
                audio = audio[:, 0]
            else:
                audio = audio.reshape(-1)
        else:
            audio = audio.reshape(-1)

        contiguous = np.ascontiguousarray(audio, dtype=np.int16)
        return _AudioChunk(
            data=contiguous.tobytes(),
            sample_count=int(contiguous.size),
            recorded_at=_utc_now(),
        )
