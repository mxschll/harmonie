"""Decoding to mono via PyAV (``features.decode_mono``)."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from harmonie.features import decode_mono
from tests.audio_fixtures import (
    encode,
    strip_flac_header,
    tone,
    write_flac,
    write_wav,
)


def _decode_within(path: Path, seconds: float):
    """Run decode_mono on a thread so a hang fails the test instead of
    stalling the run. Returns the result or re-raises."""
    box: dict = {}

    def run():
        try:
            box["result"] = decode_mono(path)
        except BaseException as exc:  # noqa: B036 - re-raised below
            box["error"] = exc

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=seconds)
    assert not t.is_alive(), f"decode_mono hung on {path.name}"
    if "error" in box:
        raise box["error"]
    return box["result"]


# ---------------------------------------------------------------------------
# Downmix and rate
# ---------------------------------------------------------------------------


def test_stereo_is_averaged_like_monomixer(tmp_path: Path):
    rate = 44100
    left = tone(rate, 0.5, 440)
    right = tone(rate, 0.5, 660)
    write_wav(tmp_path / "s.wav", rate, np.stack([left, right]))

    audio, got_rate = decode_mono(tmp_path / "s.wav")

    assert got_rate == rate
    assert audio.dtype == np.float32
    assert audio.shape == (left.size,)
    # Exact: s16 -> float is a power-of-two scale and the two-channel average
    # is exact in float32. Parity with MonoLoader is bit for bit.
    expected = (left.astype(np.float32) + right.astype(np.float32)) / 2 / 32768
    np.testing.assert_array_equal(audio, expected)


def test_mono_passes_through(tmp_path: Path):
    rate = 44100
    mono = tone(rate, 0.25, 220)
    write_wav(tmp_path / "m.wav", rate, mono[None, :])

    audio, got_rate = decode_mono(tmp_path / "m.wav")

    assert got_rate == rate
    np.testing.assert_array_equal(audio, mono.astype(np.float32) / 32768)


def test_native_rate_is_reported_not_resampled(tmp_path: Path):
    rate = 48000
    mono = tone(rate, 0.25, 220)
    write_wav(tmp_path / "48k.wav", rate, mono[None, :])

    audio, got_rate = decode_mono(tmp_path / "48k.wav")

    assert got_rate == 48000
    assert audio.size == mono.size


def test_flac_16_bit(tmp_path: Path):
    rate = 44100
    left = tone(rate, 0.5, 440)
    right = tone(rate, 0.5, 660)
    write_flac(tmp_path / "s.flac", rate, "stereo", np.stack([left, right]))

    audio, got_rate = decode_mono(tmp_path / "s.flac")

    assert got_rate == rate
    expected = (left.astype(np.float32) + right.astype(np.float32)) / 2 / 32768
    np.testing.assert_array_equal(audio, expected)


def test_flac_24_bit(tmp_path: Path):
    """24-bit FLAC decodes to s32 with the sample in the top three bytes."""
    rate = 48000
    tone24 = (tone(rate, 0.25, 440).astype(np.int32)) << 8
    encode(tmp_path / "hi.flac", "flac", rate, "mono", tone24[None, :] << 8, "s32")

    audio, got_rate = decode_mono(tmp_path / "hi.flac")

    assert got_rate == rate
    expected = (tone24.astype(np.float32) * 256) / 2**31
    np.testing.assert_allclose(audio, expected, atol=2**-23)


def test_lossy_codec_takes_the_planar_float_path(tmp_path: Path):
    """AAC decodes straight to planar float, so the resampler passes frames
    through untouched. The decoded length is the input padded to a codec
    frame; the content is lossy so only shape and rate are pinned."""
    rate = 44100
    left = tone(rate, 0.5, 440)
    right = tone(rate, 0.5, 660)
    encode(
        tmp_path / "a.m4a",
        "aac",
        rate,
        "stereo",
        np.stack([left, right]).astype(np.float32) / 32768,
        "fltp",
    )

    audio, got_rate = decode_mono(tmp_path / "a.m4a")

    assert got_rate == rate
    assert audio.dtype == np.float32
    # Encoder priming plus padding to the 1024-sample AAC frame.
    assert 0 <= audio.size - left.size < 2048


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_garbage_file_raises(tmp_path: Path):
    bogus = tmp_path / "notes.flac"
    bogus.write_text("this is not audio")
    with pytest.raises(ValueError):
        decode_mono(bogus)


def test_file_without_an_audio_stream_raises(tmp_path: Path):
    """A subtitle file is a valid container with no audio in it."""
    srt = tmp_path / "mislabelled.flac"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n")
    with pytest.raises(ValueError, match="no audio stream"):
        decode_mono(srt)


def test_empty_wav_decodes_to_empty(tmp_path: Path):
    write_wav(tmp_path / "e.wav", 44100, np.zeros((2, 0), dtype=np.int16))
    audio, rate = decode_mono(tmp_path / "e.wav")
    assert audio.size == 0
    assert rate == 44100


def test_more_than_two_channels_is_rejected(tmp_path: Path):
    """Same as MonoLoader: surround needs a real downmix, not an average."""
    rate = 48000
    channel = tone(rate, 0.25, 440)
    write_wav(tmp_path / "surround.wav", rate, np.stack([channel] * 6))
    with pytest.raises(ValueError, match="more than 2"):
        decode_mono(tmp_path / "surround.wav")


def test_damaged_stretch_is_skipped_not_fatal(tmp_path: Path, caplog):
    """MonoLoader skipped packets the decoder rejected and kept going. A file
    with a zeroed region must still yield the rest of its audio."""
    rate = 44100
    # AAC in MP4: the demuxer hands the zeroed bytes to the decoder as packets
    # it then rejects. (FLAC's parser resyncs past zeros on its own, so it
    # would not reach the skip path.)
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(rate * 2) * 0.25).astype(np.float32)
    good = tmp_path / "good.m4a"
    encode(good, "aac", rate, "stereo", np.stack([noise, noise]), "fltp")
    data = bytearray(good.read_bytes())
    mid = len(data) // 2
    data[mid : mid + 2048] = b"\0" * 2048
    damaged = tmp_path / "damaged.m4a"
    damaged.write_bytes(bytes(data))

    with caplog.at_level("WARNING", logger="harmonie.features"):
        audio, got_rate = decode_mono(damaged)

    assert got_rate == rate
    assert audio.size > noise.size // 2
    assert any("undecodable packet" in r.getMessage() for r in caplog.records)


def test_flac_with_mid_stream_parameter_change_fails_fast(tmp_path: Path):
    """Regression for the scan hang (issue #9). A FLAC whose sample rate and
    channel count change at a frame boundary sent the FFmpeg 2.8 that
    essentia bundles into an endless loop. A current FFmpeg decodes the
    second half at its new parameters, and decode_mono rejects the change."""
    a = tmp_path / "a.flac"
    b = tmp_path / "b.flac"
    channel = tone(44100, 1.0, 440)
    write_flac(a, 44100, "stereo", np.stack([channel, channel]))
    write_flac(b, 48000, "mono", tone(48000, 1.0, 440)[None, :])
    spliced = tmp_path / "spliced.flac"
    spliced.write_bytes(a.read_bytes() + strip_flac_header(b.read_bytes()))

    with pytest.raises(ValueError, match="change mid-file"):
        _decode_within(spliced, seconds=10)
