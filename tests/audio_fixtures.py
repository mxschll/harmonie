"""Generate small audio files for tests. Needs only numpy and PyAV."""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import numpy as np


def tone(rate: int, seconds: float, hz: float, amp: float = 0.5) -> np.ndarray:
    """A sine as int16 samples."""
    t = np.arange(int(rate * seconds)) / rate
    return (np.sin(2 * np.pi * hz * t) * amp * 32767).astype(np.int16)


def write_wav(path: Path, rate: int, channels: np.ndarray) -> None:
    """``channels`` is int16, shape (n_channels, n_samples)."""
    interleaved = np.ascontiguousarray(channels.T).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels.shape[0])
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(interleaved.tobytes())


def encode(
    path: Path,
    codec: str,
    rate: int,
    layout: str,
    channels: np.ndarray,
    fmt: str = "s16",
    container_format: str | None = None,
) -> None:
    """Encode PCM, shape (n_channels, n_samples), with PyAV. ``channels``
    must already be in the dtype ``fmt`` expects. ``container_format``
    overrides the one guessed from the file extension."""
    import av

    with av.open(str(path), "w", format=container_format) as container:
        stream = container.add_stream(codec, rate=rate)
        stream.codec_context.format = fmt
        stream.codec_context.layout = layout
        if fmt.endswith("p"):
            data = np.ascontiguousarray(channels)
        else:
            data = np.ascontiguousarray(channels.T).reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(data, format=fmt, layout=layout)
        frame.sample_rate = rate
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


def write_flac(
    path: Path,
    rate: int,
    layout: str,
    channels: np.ndarray,
    container_format: str | None = None,
) -> None:
    encode(
        path, "flac", rate, layout, channels.astype(np.int16), "s16", container_format
    )


def strip_flac_header(data: bytes) -> bytes:
    """Drop ``fLaC`` and the metadata blocks; keep the frames."""
    assert data[:4] == b"fLaC"
    pos = 4
    while True:
        header = data[pos]
        length = struct.unpack(">I", b"\0" + data[pos + 1 : pos + 4])[0]
        pos += 4 + length
        if header & 0x80:
            return data[pos:]


def write_spliced_flac(path: Path) -> None:
    """A FLAC whose sample rate and channel count change at a frame boundary.

    Reproduces the file shape behind issue #9: the FFmpeg 2.8 bundled with
    essentia loops forever on it. Built from two encodings of a tone with
    different parameters, the second one's header stripped.
    """
    a = tone(44100, 1.0, 440)
    first = path.with_suffix(".a.tmp")
    second = path.with_suffix(".b.tmp")
    write_flac(first, 44100, "stereo", np.stack([a, a]), container_format="flac")
    write_flac(
        second, 48000, "mono", tone(48000, 1.0, 440)[None, :], container_format="flac"
    )
    path.write_bytes(first.read_bytes() + strip_flac_header(second.read_bytes()))
    first.unlink()
    second.unlink()
