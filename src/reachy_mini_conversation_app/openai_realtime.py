import os
import logging
from typing import Any, Literal
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI
from numpy.typing import NDArray
from openai.types.realtime import (
    AudioTranscriptionParam,
    RealtimeAudioConfigParam,
    RealtimeAudioConfigInputParam,
    RealtimeAudioConfigOutputParam,
    RealtimeSessionCreateRequestParam,
)
from openai.types.realtime.realtime_audio_formats_param import AudioPCM
from openai.types.realtime.realtime_audio_input_turn_detection_param import ServerVad

from reachy_mini_conversation_app.config import PROJECT_ROOT, OPENAI_BACKEND, config, get_default_voice_for_backend
from reachy_mini_conversation_app.prompts import get_session_voice, get_session_instructions
from reachy_mini_conversation_app.base_realtime import BaseRealtimeHandler, to_realtime_tools_config
from reachy_mini_conversation_app.audio_recording import ConversationAudioRecorder
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies, get_active_tool_specs


logger = logging.getLogger(__name__)

__all__ = ["OpenaiRealtimeHandler"]


class OpenaiRealtimeHandler(BaseRealtimeHandler):
    """Realtime handler for the direct OpenAI Realtime API."""

    BACKEND_PROVIDER = OPENAI_BACKEND
    SAMPLE_RATE = 24000
    REFRESH_CLIENT_ON_RECONNECT = False
    AUDIO_INPUT_COST_PER_1M = 32.0
    AUDIO_OUTPUT_COST_PER_1M = 64.0
    TEXT_INPUT_COST_PER_1M = 4.0
    TEXT_OUTPUT_COST_PER_1M = 16.0
    IMAGE_INPUT_COST_PER_1M = 5.0

    def __init__(
        self,
        deps: ToolDependencies,
        gradio_mode: bool = False,
        instance_path: str | None = None,
        startup_voice: str | None = None,
        record_audio: bool = False,
    ) -> None:
        """Initialize OpenAI-specific credential state."""
        super().__init__(deps, gradio_mode, instance_path, startup_voice)
        self._key_source: Literal["env", "textbox"] = "env"
        self._provided_api_key: str | None = None
        self._record_audio = record_audio
        self._audio_recorder: ConversationAudioRecorder | None = None
        self._audio_recorder_failed = False

    def copy(self) -> "OpenaiRealtimeHandler":
        """Create a copy of the handler while preserving OpenAI-specific options."""
        return type(self)(
            self.deps,
            self.gradio_mode,
            self.instance_path,
            startup_voice=self._voice_override,
            record_audio=self._record_audio,
        )

    def _audio_recordings_root(self) -> Path:
        configured_root = (os.getenv("OPENAI_AUDIO_RECORDINGS_DIR") or "").strip()
        if configured_root:
            return Path(configured_root).expanduser()
        return PROJECT_ROOT / "recordings" / "openai"

    def _get_audio_recorder(self) -> ConversationAudioRecorder | None:
        if not self._record_audio:
            return None
        if self._audio_recorder is not None:
            return self._audio_recorder
        if self._audio_recorder_failed:
            return None

        try:
            self._audio_recorder = ConversationAudioRecorder(
                self._audio_recordings_root(),
                sample_rate=self.SAMPLE_RATE,
            )
            logger.info("OpenAI audio recording directory: %s", self._audio_recorder.run_dir)
        except Exception as exc:
            self._audio_recorder_failed = True
            logger.warning("OpenAI audio recording disabled; failed to initialize recorder: %s", exc)
            return None
        return self._audio_recorder

    def _record_sent_input_audio(self, audio_frame: NDArray[np.int16]) -> None:
        recorder = self._get_audio_recorder()
        if recorder is not None:
            recorder.buffer_sent_input_audio(audio_frame)

    def _start_recorded_user_audio_turn(self) -> None:
        recorder = self._get_audio_recorder()
        if recorder is not None:
            recorder.start_user_turn()

    def _finish_recorded_user_audio_turn(self, transcript: str | None = None) -> None:
        if self._audio_recorder is not None:
            self._audio_recorder.finish_user_turn(transcript=transcript)

    def _record_received_assistant_audio(self, audio_frame: NDArray[np.int16]) -> None:
        recorder = self._get_audio_recorder()
        if recorder is not None:
            recorder.append_assistant_audio(audio_frame)

    def _finish_recorded_assistant_audio_turn(self) -> None:
        if self._audio_recorder is not None:
            self._audio_recorder.finish_assistant_turn()

    def _close_audio_recording(self) -> None:
        if self._audio_recorder is not None:
            self._audio_recorder.close()

    async def _prepare_startup_credentials(self) -> None:
        """Collect an OpenAI API key from Gradio input when needed."""
        openai_api_key = config.OPENAI_API_KEY
        if not self.gradio_mode or openai_api_key:
            return

        await self.wait_for_args()  # type: ignore[no-untyped-call]
        args = list(self.latest_args)
        textbox_api_key = args[3] if len(args) > 3 and len(args[3]) > 0 else None
        if textbox_api_key is not None:
            self._key_source = "textbox"
            self._provided_api_key = textbox_api_key

    def _persist_credentials_if_needed(self) -> None:
        """Persist a textbox-provided OpenAI API key into the instance `.env`."""
        try:
            if not self.gradio_mode:
                logger.warning("Not in Gradio mode; skipping OpenAI API key persistence.")
                return

            if self._key_source != "textbox":
                logger.info("OpenAI API key not provided via textbox; skipping persistence.")
                return

            key = (self._provided_api_key or "").strip()
            if not key:
                logger.warning("No OpenAI API key provided via textbox; skipping persistence.")
                return
            if self.instance_path is None:
                logger.warning("Instance path is None; cannot persist OpenAI API key.")
                return

            # Update the current process environment for downstream consumers.
            try:
                import os

                os.environ["OPENAI_API_KEY"] = key
            except Exception:  # best-effort
                pass

            target_dir = Path(self.instance_path)
            env_path = target_dir / ".env"
            if env_path.exists():
                # Respect existing user configuration.
                logger.info(".env already exists at %s; not overwriting.", env_path)
                return

            example_path = target_dir / ".env.example"
            content_lines: list[str] = []
            if example_path.exists():
                try:
                    content = example_path.read_text(encoding="utf-8")
                    content_lines = content.splitlines()
                except Exception as e:
                    logger.warning("Failed to read .env.example at %s: %s", example_path, e)

            replaced = False
            for i, line in enumerate(content_lines):
                if line.strip().startswith("OPENAI_API_KEY="):
                    content_lines[i] = f"OPENAI_API_KEY={key}"
                    replaced = True
                    break
            if not replaced:
                content_lines.append(f"OPENAI_API_KEY={key}")

            final_text = "\n".join(content_lines) + "\n"
            env_path.write_text(final_text, encoding="utf-8")
            logger.info("Created %s and stored OPENAI_API_KEY for future runs.", env_path)
        except Exception as e:
            # Never crash the app for QoL persistence; just log.
            logger.warning("Could not persist OPENAI_API_KEY to .env: %s", e)

    def _get_session_instructions(self) -> str:
        """Return OpenAI session instructions."""
        return get_session_instructions(self.instance_path)

    def _get_session_voice(self, default: str | None = None) -> str:
        """Return the configured OpenAI session voice."""
        return get_session_voice(default)

    def _get_active_tool_specs(self) -> list[dict[str, Any]]:
        """Return active tool specs for the current session dependencies."""
        return get_active_tool_specs(self.deps)

    def _get_session_config(self, tool_specs: list[dict[str, Any]]) -> RealtimeSessionCreateRequestParam:
        """Return the OpenAI Realtime session config."""
        return RealtimeSessionCreateRequestParam(
            type="realtime",
            instructions=self._get_session_instructions(),
            audio=RealtimeAudioConfigParam(
                input=RealtimeAudioConfigInputParam(
                    format=AudioPCM(type="audio/pcm", rate=24000),
                    transcription=AudioTranscriptionParam(
                        model="gpt-4o-transcribe",
                        language=config.REALTIME_TRANSCRIPTION_LANGUAGE,
                    ),
                    turn_detection=ServerVad(type="server_vad", interrupt_response=True),
                ),
                output=RealtimeAudioConfigOutputParam(
                    format=AudioPCM(type="audio/pcm", rate=24000),
                    voice=self.get_current_voice(),
                ),
            ),
            tools=to_realtime_tools_config(tool_specs),
            tool_choice="auto",
        )

    async def get_available_voices(self) -> list[str]:
        """Try to discover available voices for the configured OpenAI realtime model.

        Attempts to retrieve model metadata from the OpenAI Models API and look
        for any keys that might contain voice names. Falls back to a curated
        list known to work with realtime if discovery fails.
        """
        fallback = await super().get_available_voices()
        try:
            model = await self.client.models.retrieve(config.MODEL_NAME)
            raw = None
            for attr in ("model_dump", "to_dict"):
                fn = getattr(model, attr, None)
                if callable(fn):
                    try:
                        raw = fn()
                        break
                    except Exception:
                        pass
            if raw is None:
                try:
                    raw = dict(model)
                except Exception:
                    raw = None

            candidates: set[str] = set()

            def _collect(obj: object) -> None:
                try:
                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            key_lower = str(key).lower()
                            if "voice" in key_lower and isinstance(value, (list, tuple)):
                                for item in value:
                                    if isinstance(item, str):
                                        candidates.add(item)
                                    elif isinstance(item, dict) and isinstance(item.get("name"), str):
                                        candidates.add(item["name"])
                            else:
                                _collect(value)
                    elif isinstance(obj, (list, tuple)):
                        for item in obj:
                            _collect(item)
                except Exception:
                    pass

            if isinstance(raw, dict):
                _collect(raw)

            voices = sorted(candidates) if candidates else fallback
            default_voice = get_default_voice_for_backend(self.BACKEND_PROVIDER)
            if default_voice not in voices:
                voices = [default_voice, *[voice for voice in voices if voice != default_voice]]
            return voices
        except Exception:
            return fallback

    async def _build_realtime_client(self) -> AsyncOpenAI:
        """Build the OpenAI realtime SDK client."""
        self._realtime_connect_query = {}
        resolved_api_key = (self._provided_api_key or config.OPENAI_API_KEY or "").strip()
        if not resolved_api_key:
            # In headless console mode, LocalStream blocks startup until the key is provided.
            # Unit tests may invoke this handler directly with a stubbed client.
            logger.warning("OPENAI_API_KEY missing. Proceeding with a placeholder (tests/offline).")
            resolved_api_key = "DUMMY"
        return AsyncOpenAI(api_key=resolved_api_key)
