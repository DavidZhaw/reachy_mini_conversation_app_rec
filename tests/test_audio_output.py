import numpy as np

from reachy_mini_conversation_app.audio_output import Pcm16PeakNormalizer, normalize_pcm16_peak


def test_normalize_pcm16_peak_amplifies_quiet_audio() -> None:
    """Quiet PCM frames should be amplified up to the target peak."""
    audio = np.array([[-1000, 0, 1000]], dtype=np.int16)

    normalized = normalize_pcm16_peak(audio)

    assert normalized.dtype == np.int16
    assert normalized.shape == audio.shape
    assert int(np.max(np.abs(normalized.astype(np.int32)))) == int(np.iinfo(np.int16).max * 0.98)


def test_normalize_pcm16_peak_leaves_silence_unchanged() -> None:
    """Silent PCM frames should not be copied or amplified."""
    audio = np.zeros((1, 128), dtype=np.int16)

    normalized = normalize_pcm16_peak(audio)

    assert normalized is audio


def test_normalize_pcm16_peak_leaves_pause_noise_unchanged() -> None:
    """Very quiet nonzero pause noise should not be amplified into hiss."""
    audio = np.array([[0, 8, -12, 4, -6, 10, -9, 3]], dtype=np.int16)

    normalized = normalize_pcm16_peak(audio)

    assert normalized is audio


def test_normalize_pcm16_peak_does_not_reduce_loud_audio() -> None:
    """Already-loud PCM frames should not be attenuated."""
    audio = np.array([[-32768, 1000, 32767]], dtype=np.int16)

    normalized = normalize_pcm16_peak(audio)

    np.testing.assert_array_equal(normalized, audio)


def test_pcm16_peak_normalizer_smooths_gain_increases() -> None:
    """Stateful normalization should avoid sudden per-snippet gain jumps."""
    normalizer = Pcm16PeakNormalizer(gain_release=0.1)
    audio = np.array([[-1000, 0, 1000]], dtype=np.int16)

    normalized = normalizer.process(audio)

    assert normalizer.gain < 4.5
    assert int(np.max(np.abs(normalized.astype(np.int32)))) < 4500


def test_pcm16_peak_normalizer_lowers_gain_for_louder_snippets() -> None:
    """Stateful normalization should react quickly when the next snippet is louder."""
    normalizer = Pcm16PeakNormalizer(gain_release=1.0)
    quiet = np.array([[-1000, 0, 1000]], dtype=np.int16)
    louder = np.array([[-16000, 0, 16000]], dtype=np.int16)

    normalizer.process(quiet)
    normalized = normalizer.process(louder)

    assert int(np.max(np.abs(normalized.astype(np.int32)))) <= int(np.iinfo(np.int16).max * 0.98)
