import wave

def generate_silent_wav(filename, duration=5, sample_rate=16000):
    """Generates a silent WAV file of given duration and sample rate."""
    with wave.open(filename, "w") as f:
        f.setnchannels(1)  # Mono
        f.setsampwidth(2)  # 16-bit
        f.setframerate(sample_rate)
        f.writeframes(b'\x00\x00' * duration * sample_rate)

generate_silent_wav("silent_audio.wav", duration=60)