"""Tests for command-line argument parsing."""

import sys

import pytest

from reachy_mini_conversation_app.utils import parse_args


def test_parse_args_audio_recording_defaults_to_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Realtime audio recording should be opt-in from the CLI."""
    monkeypatch.setattr(sys, "argv", ["reachy-mini-conversation-app"])

    args, _unknown = parse_args()

    assert args.record_audio is False
    assert args.diarize_audio is False
    assert args.normalize_output_audio is False


def test_parse_args_accepts_audio_recording_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI should expose an opt-in flag for realtime audio recording."""
    monkeypatch.setattr(sys, "argv", ["reachy-mini-conversation-app", "--record-audio"])

    args, _unknown = parse_args()

    assert args.record_audio is True


def test_parse_args_accepts_audio_diarization_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI should expose an opt-in flag for recorded user-audio diarization."""
    monkeypatch.setattr(sys, "argv", ["reachy-mini-conversation-app", "--diarize-audio"])

    args, _unknown = parse_args()

    assert args.diarize_audio is True


def test_parse_args_accepts_audio_diarization_speaker_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI should accept optional speaker names for audio diarization."""
    monkeypatch.setattr(sys, "argv", ["reachy-mini-conversation-app", "--diarize-audio", "bob", "alice"])

    args, _unknown = parse_args()

    assert args.diarize_audio == ["bob", "alice"]


def test_parse_args_accepts_audio_diarization_json_speaker_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI should accept speaker names as a JSON array."""
    monkeypatch.setattr(sys, "argv", ["reachy-mini-conversation-app", "--diarize-audio", '["bob","alice"]'])

    args, _unknown = parse_args()

    assert args.diarize_audio == ["bob", "alice"]


def test_parse_args_accepts_output_audio_normalization_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI should expose opt-in output audio normalization."""
    monkeypatch.setattr(sys, "argv", ["reachy-mini-conversation-app", "--normalize-output-audio"])

    args, _unknown = parse_args()

    assert args.normalize_output_audio is True
