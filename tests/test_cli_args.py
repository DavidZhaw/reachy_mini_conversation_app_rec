"""Tests for command-line argument parsing."""

import sys

import pytest

from reachy_mini_conversation_app.utils import parse_args


def test_parse_args_openai_audio_recording_defaults_to_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI audio recording should be opt-in from the CLI."""
    monkeypatch.setattr(sys, "argv", ["reachy-mini-conversation-app"])

    args, _unknown = parse_args()

    assert args.record_openai_audio is False


def test_parse_args_accepts_openai_audio_recording_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI should expose an opt-in flag for OpenAI audio recording."""
    monkeypatch.setattr(sys, "argv", ["reachy-mini-conversation-app", "--record-openai-audio"])

    args, _unknown = parse_args()

    assert args.record_openai_audio is True
