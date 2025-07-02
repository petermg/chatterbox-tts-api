import requests
import sounddevice as sd
import numpy as np

API_URL = "http://localhost:4123/audio/speech/stream"


def parse_wav_header(header_bytes):
    if len(header_bytes) < 44:
        raise ValueError("Header too short for WAV file")
    channels = int.from_bytes(header_bytes[22:24], "little")
    sample_rate = int.from_bytes(header_bytes[24:28], "little")
    bits_per_sample = int.from_bytes(header_bytes[34:36], "little")
    audio_format = int.from_bytes(header_bytes[20:22], "little")
    return channels, sample_rate, bits_per_sample, audio_format

def stream_and_play_sd(text, voice="alloy", streaming_quality="balanced"):
    json_payload = {
        "input": text,
        "voice": voice,
        "streaming_quality": streaming_quality  # <--- Key line!
    }
    with requests.post(API_URL, json=json_payload, stream=True) as r:
        r.raise_for_status()
        it = r.iter_content(chunk_size=4096)

        wav_header = b''
        while len(wav_header) < 44:
            try:
                chunk = next(it)
            except StopIteration:
                raise RuntimeError("No WAV header received")
            needed = 44 - len(wav_header)
            wav_header += chunk[:needed]
            remainder = chunk[needed:] if needed < len(chunk) else b''

        channels, sr, bits_per_sample, audio_format = parse_wav_header(wav_header)
        if audio_format == 1 and bits_per_sample == 16:
            dtype = 'int16'
        elif audio_format == 3 and bits_per_sample == 32:
            dtype = 'float32'
        else:
            raise RuntimeError(f"Unsupported audio format: {audio_format} with {bits_per_sample} bits")

        print(f"Sample Rate: {sr}, Channels: {channels}, Bits: {bits_per_sample}, Format: {audio_format}, dtype: {dtype}")

        def audio_gen():
            if remainder:
                data = np.frombuffer(remainder, dtype=dtype)
                if channels > 1:
                    data = data.reshape(-1, channels)
                yield data
            for chunk in it:
                if chunk:
                    data = np.frombuffer(chunk, dtype=dtype)
                    if channels > 1:
                        data = data.reshape(-1, channels)
                    yield data

        print(f"Playing live... (quality={streaming_quality})")
        stream = sd.OutputStream(
            samplerate=sr,
            channels=channels,
            dtype=dtype,
            blocksize=1024
        )
        stream.start()
        try:
            for audio_chunk in audio_gen():
                stream.write(audio_chunk)
        finally:
            stream.stop()
            stream.close()
        print("Done.")

if __name__ == "__main__":
    # ---- PLAY WITH THIS LINE ----
    # Options: "fast", "balanced", "high"
    streaming_quality = "fast"
    text = (
        "Hello world! This is a streaming test. "
        "This plays in real-time as it streams! "
        "The mountains are high and covered in snow. "
        "The storm is strong and windy. "
        "The sun rises over the African plains. "
        "The sun sets behind the ocean. "
        "The sun rises in the east and the sun sets in the West. "
        "Tonight there is not a full moon."
    )
    stream_and_play_sd(text, voice="alloy", streaming_quality=streaming_quality)
