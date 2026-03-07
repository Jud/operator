#!/usr/bin/env python3
"""Generate audio feedback tone .caf files for Operator.

Each tone is a short synthesized audio cue:
- listening: soft rising tone (300Hz -> 500Hz, 0.15s)
- processing: brief double-tick (two short 800Hz pulses, 0.2s total)
- delivered: low pleasant chime (400Hz with harmonic at 800Hz, 0.25s)
- error: descending two-tone (600Hz -> 400Hz, 0.3s)
- pending: subtle notification (500Hz soft pulse, 0.15s)
"""

import struct
import math
import subprocess
import os
import sys
import tempfile

SAMPLE_RATE = 44100
AMPLITUDE = 0.4  # moderate volume


def write_wav(filepath: str, samples: list[float]) -> None:
    """Write mono 16-bit PCM WAV file."""
    num_samples = len(samples)
    data_size = num_samples * 2  # 16-bit = 2 bytes per sample
    with open(filepath, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))  # chunk size
        f.write(struct.pack("<H", 1))  # PCM format
        f.write(struct.pack("<H", 1))  # mono
        f.write(struct.pack("<I", SAMPLE_RATE))
        f.write(struct.pack("<I", SAMPLE_RATE * 2))  # byte rate
        f.write(struct.pack("<H", 2))  # block align
        f.write(struct.pack("<H", 16))  # bits per sample
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        for s in samples:
            clamped = max(-1.0, min(1.0, s))
            f.write(struct.pack("<h", int(clamped * 32767)))


def fade_envelope(samples: list[float], fade_in_ms: int = 5, fade_out_ms: int = 15) -> list[float]:
    """Apply fade in/out envelope to avoid clicks."""
    fade_in = int(SAMPLE_RATE * fade_in_ms / 1000)
    fade_out = int(SAMPLE_RATE * fade_out_ms / 1000)
    result = list(samples)
    for i in range(min(fade_in, len(result))):
        result[i] *= i / fade_in
    for i in range(min(fade_out, len(result))):
        idx = len(result) - 1 - i
        result[idx] *= i / fade_out
    return result


def sine(freq: float, duration: float, amplitude: float = AMPLITUDE) -> list[float]:
    """Generate sine wave samples."""
    n = int(SAMPLE_RATE * duration)
    return [amplitude * math.sin(2 * math.pi * freq * i / SAMPLE_RATE) for i in range(n)]


def chirp(freq_start: float, freq_end: float, duration: float, amplitude: float = AMPLITUDE) -> list[float]:
    """Generate frequency sweep (linear chirp)."""
    n = int(SAMPLE_RATE * duration)
    samples = []
    for i in range(n):
        t = i / SAMPLE_RATE
        freq = freq_start + (freq_end - freq_start) * (t / duration)
        samples.append(amplitude * math.sin(2 * math.pi * freq * t))
    return samples


def silence(duration: float) -> list[float]:
    """Generate silence."""
    return [0.0] * int(SAMPLE_RATE * duration)


def generate_listening() -> list[float]:
    """Soft rising tone -- 'I'm listening'."""
    return fade_envelope(chirp(300, 500, 0.15, AMPLITUDE * 0.7), fade_in_ms=5, fade_out_ms=20)


def generate_processing() -> list[float]:
    """Brief double-tick -- 'Got it, processing'."""
    tick1 = sine(800, 0.04, AMPLITUDE * 0.6)
    gap = silence(0.06)
    tick2 = sine(800, 0.04, AMPLITUDE * 0.6)
    samples = tick1 + gap + tick2
    return fade_envelope(samples, fade_in_ms=2, fade_out_ms=5)


def generate_delivered() -> list[float]:
    """Low pleasant chime -- 'Sent'."""
    n = int(SAMPLE_RATE * 0.25)
    samples = []
    for i in range(n):
        t = i / SAMPLE_RATE
        # fundamental at 400Hz + harmonic at 800Hz (half amplitude)
        s = AMPLITUDE * 0.5 * math.sin(2 * math.pi * 400 * t)
        s += AMPLITUDE * 0.25 * math.sin(2 * math.pi * 800 * t)
        # gentle exponential decay
        s *= math.exp(-3.0 * t)
        samples.append(s)
    return fade_envelope(samples, fade_in_ms=3, fade_out_ms=30)


def generate_error() -> list[float]:
    """Descending two-tone -- 'Something went wrong'."""
    tone1 = sine(600, 0.12, AMPLITUDE * 0.6)
    gap = silence(0.03)
    tone2 = sine(400, 0.15, AMPLITUDE * 0.6)
    samples = tone1 + gap + tone2
    return fade_envelope(samples, fade_in_ms=3, fade_out_ms=20)


def generate_pending() -> list[float]:
    """Subtle notification pulse -- 'Agents have things to say'."""
    n = int(SAMPLE_RATE * 0.15)
    samples = []
    for i in range(n):
        t = i / SAMPLE_RATE
        # soft 500Hz with quick decay
        s = AMPLITUDE * 0.4 * math.sin(2 * math.pi * 500 * t)
        s *= math.exp(-5.0 * t)
        samples.append(s)
    return fade_envelope(samples, fade_in_ms=3, fade_out_ms=10)


def main() -> None:
    resources_dir = os.path.join(os.path.dirname(__file__), "Sources", "Resources")
    os.makedirs(resources_dir, exist_ok=True)

    tones = {
        "listening": generate_listening,
        "processing": generate_processing,
        "delivered": generate_delivered,
        "error": generate_error,
        "pending": generate_pending,
    }

    for name, generator in tones.items():
        samples = generator()
        wav_path = os.path.join(tempfile.gettempdir(), f"{name}.wav")
        caf_path = os.path.join(resources_dir, f"{name}.caf")

        write_wav(wav_path, samples)

        result = subprocess.run(
            ["afconvert", "-f", "caff", "-d", "LEI16", wav_path, caf_path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Error converting {name}: {result.stderr}", file=sys.stderr)
            sys.exit(1)

        os.remove(wav_path)
        print(f"Generated {caf_path}")

    print("All tones generated successfully.")


if __name__ == "__main__":
    main()
