"""Audio output processing helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


_INT16_MAX = np.iinfo(np.int16).max
_INT16_MIN = np.iinfo(np.int16).min


class Pcm16PeakNormalizer:
    """Peak-normalize PCM frames with smoothed gain between snippets."""

    def __init__(
        self,
        target_peak: float = 0.98,
        min_rms: float = 200.0,
        max_gain: float = 12.0,
        gain_attack: float = 0.75,
        gain_release: float = 0.08,
        silence_release: float = 0.15,
    ) -> None:
        """Initialize normalizer settings."""
        self.target_peak = target_peak
        self.min_rms = min_rms
        self.max_gain = max_gain
        self.gain_attack = gain_attack
        self.gain_release = gain_release
        self.silence_release = silence_release
        self._gain = 1.0

    @property
    def gain(self) -> float:
        """Return the current smoothed gain."""
        return self._gain

    def process(self, audio_frame: NDArray[np.int16]) -> NDArray[np.int16]:
        """Normalize an int16 PCM frame without boosting silence or pause noise."""
        if audio_frame.size == 0:
            return audio_frame

        audio_int32 = audio_frame.astype(np.int32)
        rms = float(np.sqrt(np.mean(np.square(audio_int32, dtype=np.float64))))
        if rms < self.min_rms:
            self._gain += (1.0 - self._gain) * self.silence_release
            return audio_frame

        peak = int(np.max(np.abs(audio_int32)))
        if peak == 0:
            return audio_frame

        target = int(_INT16_MAX * self.target_peak)
        desired_gain = min(self.max_gain, max(1.0, target / peak))
        smoothing = self.gain_attack if desired_gain < self._gain else self.gain_release
        self._gain += (desired_gain - self._gain) * smoothing

        gain = min(self._gain, target / peak)
        if gain <= 1.0:
            return audio_frame

        return _apply_pcm16_gain(audio_frame, gain)


def _apply_pcm16_gain(audio_frame: NDArray[np.int16], gain: float) -> NDArray[np.int16]:
    """Apply a gain multiplier to int16 PCM samples."""
    normalized = np.rint(audio_frame.astype(np.float32) * gain)
    normalized = np.clip(normalized, _INT16_MIN, _INT16_MAX)
    return normalized.astype(np.int16)


def normalize_pcm16_peak(
    audio_frame: NDArray[np.int16],
    target_peak: float = 0.98,
    min_rms: float = 200.0,
) -> NDArray[np.int16]:
    """Peak-normalize one int16 PCM frame without boosting silence or pause noise."""
    return Pcm16PeakNormalizer(
        target_peak=target_peak,
        min_rms=min_rms,
        max_gain=float("inf"),
        gain_release=1.0,
    ).process(audio_frame)
