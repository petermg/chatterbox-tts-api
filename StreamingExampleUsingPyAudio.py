import requests
import pyaudio
import numpy as np

API_URL = "http://localhost:4123/audio/speech/stream"
TEXT = "Hello world! This is a streaming test. This plays in real-time as it streams! The mountains are high and covered in snow. The storm is strong and windy. The sun rises over the african plains. The sun sets behind the ocean. The sun rises in the east and the sun sets in the West. Tonight there is not a full mooon."
VOICE = "alloy"

json_payload = {"input": TEXT, "voice": VOICE}

def parse_wav_header(header_bytes):
    if len(header_bytes) < 44:
        raise ValueError("Header too short for WAV file")
    channels = int.from_bytes(header_bytes[22:24], "little")
    sample_rate = int.from_bytes(header_bytes[24:28], "little")
    bits_per_sample = int.from_bytes(header_bytes[34:36], "little")
    audio_format = int.from_bytes(header_bytes[20:22], "little")
    return channels, sample_rate, bits_per_sample, audio_format

def stream_and_play_pyaudio():
    with requests.post(API_URL, json=json_payload, stream=True) as r:
        r.raise_for_status()
        it = r.iter_content(chunk_size=4096)

        # Assemble the WAV header from the first bytes of the stream
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
            np_dtype = np.int16
            pa_format = pyaudio.paInt16
        elif audio_format == 3 and bits_per_sample == 32:
            np_dtype = np.float32
            pa_format = pyaudio.paFloat32
        else:
            raise RuntimeError(f"Unsupported audio format: {audio_format} with {bits_per_sample} bits")

        print(f"Sample Rate: {sr}, Channels: {channels}, Bits: {bits_per_sample}, Format: {audio_format}, dtype: {np_dtype}")

        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pa_format,
            channels=channels,
            rate=sr,
            output=True
        )

        def audio_gen():
            if remainder:
                data = np.frombuffer(remainder, dtype=np_dtype)
                if channels > 1:
                    data = data.reshape(-1, channels)
                yield data
            for chunk in it:
                if chunk:
                    data = np.frombuffer(chunk, dtype=np_dtype)
                    if channels > 1:
                        data = data.reshape(-1, channels)
                    yield data

        print("Playing live...")
        try:
            for audio_chunk in audio_gen():
                # CONVERT to bytes for PyAudio!
                stream.write(audio_chunk.tobytes())
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()
        print("Done.")


if __name__ == "__main__":
    stream_and_play_pyaudio()
