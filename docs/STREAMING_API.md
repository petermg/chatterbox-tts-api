# Streaming TTS API Documentation

## 🎵 Overview

The Chatterbox TTS API supports real-time audio streaming, allowing clients to receive audio data as it's generated rather than waiting for complete processing. This significantly reduces perceived latency and improves user experience, especially for longer texts.

## ✨ Key Benefits

- **Lower Latency**: Start receiving audio before full generation is complete
- **Better User Experience**: Perceived faster response times for long texts
- **Resource Efficiency**: Lower memory usage as chunks are processed individually
- **Real-time Processing**: Audio generation happens progressively
- **Interruption Support**: Can potentially stop generation mid-stream if needed
- **Memory Optimization**: Automatic cleanup of processed chunks

## 🚀 Streaming Endpoints

### Basic Streaming

**POST** `/audio/speech/stream`

Generate and stream speech audio in real-time using the configured voice sample.

**Request Body (JSON):**

```json
{
  "input": "Text to convert to speech",
  "exaggeration": 0.7,
  "cfg_weight": 0.4,
  "temperature": 0.9,
  "streaming_chunk_size": 200,
  "streaming_strategy": "sentence"
}
```

### Streaming with Voice Upload

**POST** `/audio/speech/stream/upload`

Generate and stream speech audio with optional custom voice file upload.

**Request (Multipart Form):**

- `input` (string): Text to convert to speech
- `voice_file` (file, optional): Custom voice sample file
- `exaggeration` (float, optional): Emotion intensity (0.25-2.0)
- `cfg_weight` (float, optional): Pace control (0.0-1.0)
- `temperature` (float, optional): Sampling randomness (0.05-5.0)
- `streaming_chunk_size` (int, optional): Characters per streaming chunk
- `streaming_strategy` (string, optional): Chunking strategy

## 🎛️ Streaming Parameters

### Core TTS Parameters

| Parameter      | Type  | Range    | Default | Description         |
| -------------- | ----- | -------- | ------- | ------------------- |
| `exaggeration` | float | 0.25-2.0 | 0.5     | Emotion intensity   |
| `cfg_weight`   | float | 0.0-1.0  | 0.5     | Pace control        |
| `temperature`  | float | 0.05-5.0 | 0.8     | Sampling randomness |

### Streaming-Specific Parameters

| Parameter               | Type   | Options                          | Default    | Description                        |
| ----------------------- | ------ | -------------------------------- | ---------- | ---------------------------------- |
| `streaming_chunk_size`  | int    | 50-500                           | 200        | Characters per streaming chunk     |
| `streaming_strategy`    | string | sentence, paragraph, fixed, word | "sentence" | How to break up text for streaming |
| `streaming_buffer_size` | int    | 1-10                             | 3          | Number of chunks to buffer         |
| `streaming_quality`     | string | fast, balanced, high             | "balanced" | Speed vs quality trade-off         |

## 📝 Streaming Strategies

### Sentence Strategy (Default)

```json
{
  "streaming_strategy": "sentence",
  "streaming_chunk_size": 200
}
```

- Splits at sentence boundaries (`.`, `!`, `?`)
- Respects sentence integrity
- Good balance of latency and naturalness
- **Best for**: General use, reading content

### Paragraph Strategy

```json
{
  "streaming_strategy": "paragraph",
  "streaming_chunk_size": 400
}
```

- Splits at paragraph breaks (`\n\n`, double line breaks)
- Maintains paragraph context
- Longer chunks, more natural flow
- **Best for**: Articles, stories, structured content

### Fixed Strategy

```json
{
  "streaming_strategy": "fixed",
  "streaming_chunk_size": 150
}
```

- Fixed character count chunks
- Predictable timing
- May break mid-sentence
- **Best for**: Consistent streaming timing, testing

### Word Strategy

```json
{
  "streaming_strategy": "word",
  "streaming_chunk_size": 100
}
```

- Splits at word boundaries
- Very fine-grained streaming
- Maximum responsiveness
- **Best for**: Real-time chat, interactive applications

## 🎯 Quality vs Speed Settings

### Fast Mode

```json
{
  "streaming_quality": "fast",
  "streaming_chunk_size": 100,
  "streaming_strategy": "word"
}
```

- Smaller chunks for faster initial response
- Lower quality synthesis parameters
- **Use case**: Chat applications, real-time feedback

### Balanced Mode (Default)

```json
{
  "streaming_quality": "balanced",
  "streaming_chunk_size": 200,
  "streaming_strategy": "sentence"
}
```

- Good balance of speed and quality
- Sentence-aware chunking
- **Use case**: General applications

### High Quality Mode

```json
{
  "streaming_quality": "high",
  "streaming_chunk_size": 300,
  "streaming_strategy": "paragraph"
}
```

- Larger chunks for better context
- Higher quality synthesis
- **Use case**: Audiobooks, professional content

## 💻 Usage Examples

### Basic cURL Examples

**Simple Streaming:**

```bash
curl -X POST http://localhost:4123/v1/audio/speech/stream \
  -H "Content-Type: application/json" \
  -d '{"input": "This will stream as it generates!"}' \
  --output streaming.wav
```

**Advanced Streaming with Custom Settings:**

```bash
curl -X POST http://localhost:4123/v1/audio/speech/stream \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Long text that will be streamed efficiently...",
    "exaggeration": 0.8,
    "streaming_strategy": "sentence",
    "streaming_chunk_size": 150,
    "streaming_quality": "balanced"
  }' \
  --output advanced_stream.wav
```

**Real-time Playback:**

```bash
curl -X POST http://localhost:4123/v1/audio/speech/stream \
  -H "Content-Type: application/json" \
  -d '{"input": "Play this as it streams!", "streaming_quality": "fast"}' \
  | ffplay -f wav -i pipe:0 -autoexit -nodisp
```

### Python Examples

#### Basic Streaming

```python
import requests

response = requests.post(
    "http://localhost:4123/v1/audio/speech/stream",
    json={
        "input": "This streams as it's generated!",
        "streaming_strategy": "sentence",
        "streaming_chunk_size": 200
    },
    stream=True
)

with open("streaming_output.wav", "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)
            print(f"Received chunk: {len(chunk)} bytes")
```

#### Advanced Streaming with Progress

```python
import requests
import threading
import time

def stream_with_progress(text, **params):
    """Stream TTS with real-time progress monitoring"""

    # Start streaming request
    response = requests.post(
        "http://localhost:4123/v1/audio/speech/stream",
        json={"input": text, **params},
        stream=True
    )

    # Monitor progress in separate thread
    def monitor_progress():
        while True:
            try:
                progress = requests.get("http://localhost:4123/v1/status/progress").json()
                if progress.get("is_processing"):
                    print(f"Progress: {progress.get('progress_percentage', 0):.1f}%")
                    print(f"Step: {progress.get('current_step', '')}")
                else:
                    break
                time.sleep(0.5)
            except:
                break

    progress_thread = threading.Thread(target=monitor_progress)
    progress_thread.start()

    # Stream audio
    with open("streaming_output.wav", "wb") as f:
        total_bytes = 0
        for chunk in response.iter_content(chunk_size=4096):
            if chunk:
                f.write(chunk)
                total_bytes += len(chunk)
                print(f"Streamed {total_bytes:,} bytes")

    progress_thread.join()
    print("Streaming complete!")

# Usage
stream_with_progress(
    "This is a long text that demonstrates streaming with progress monitoring.",
    streaming_strategy="sentence",
    streaming_chunk_size=180,
    streaming_quality="balanced"
)
```

#### Real-time Playback with pyaudio (DO NOT USE THIS, IT'S WRONG, IT'S FROM THE ORIGINAL DEV AND IS INCORRECT. Instead use either [THIS SCRIPT](https://github.com/petermg/chatterbox-tts-api/blob/main/StreamingExampleUsingPyAudio.py) for PyAudio or [THIS SCRIPT](https://github.com/petermg/chatterbox-tts-api/blob/main/StreamingExampleUsingSoundDevice.py) for SoundDevice.)

```python
import requests
import pyaudio
import wave
import io
import threading

def stream_and_play_realtime(text, **params):
    """Stream TTS and play audio in real-time using pyaudio"""

    # Audio settings
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 22050  # Adjust based on your TTS model

    # Initialize PyAudio
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        output=True,
        frames_per_buffer=CHUNK
    )

    # Start streaming request
    response = requests.post(
        "http://localhost:4123/v1/audio/speech/stream",
        json={"input": text, **params},
        stream=True
    )

    # Buffer for WAV processing
    audio_buffer = io.BytesIO()
    header_processed = False

    try:
        for chunk in response.iter_content(chunk_size=4096):
            if chunk:
                audio_buffer.write(chunk)

                # Skip WAV header for first chunk
                if not header_processed:
                    audio_buffer.seek(44)  # Skip WAV header
                    header_processed = True

                # Read and play audio data
                audio_buffer.seek(-len(chunk), 1)
                audio_data = audio_buffer.read()
                if len(audio_data) >= CHUNK:
                    stream.write(audio_data[:CHUNK])

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

# Usage
stream_and_play_realtime(
    "This plays in real-time as it streams!",
    streaming_quality="fast",
    streaming_strategy="word"
)
```

### JavaScript/TypeScript Examples

#### Basic Streaming

```typescript
async function streamTTS(text: string, options: any = {}) {
  const response = await fetch('/v1/audio/speech/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      input: text,
      streaming_strategy: 'sentence',
      streaming_chunk_size: 200,
      ...options,
    }),
  });

  const reader = response.body?.getReader();
  const chunks: Uint8Array[] = [];

  while (true) {
    const { done, value } = await reader!.read();
    if (done) break;

    chunks.push(value);
    console.log(`Received chunk: ${value.length} bytes`);
  }

  // Combine chunks into final audio
  const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const audioData = new Uint8Array(totalLength);
  let offset = 0;

  for (const chunk of chunks) {
    audioData.set(chunk, offset);
    offset += chunk.length;
  }

  return audioData;
}
```

#### Real-time Audio Playback

```typescript
async function streamAndPlayTTS(text: string) {
  const audioContext = new AudioContext();
  const response = await fetch('/v1/audio/speech/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      input: text,
      streaming_quality: 'fast',
    }),
  });

  const reader = response.body?.getReader();
  let audioBuffer = new Uint8Array();

  while (true) {
    const { done, value } = await reader!.read();
    if (done) break;

    // Append new chunk
    const newBuffer = new Uint8Array(audioBuffer.length + value.length);
    newBuffer.set(audioBuffer);
    newBuffer.set(value, audioBuffer.length);
    audioBuffer = newBuffer;

    // Try to decode and play if we have enough data
    if (audioBuffer.length > 8192) {
      try {
        const audioData = await audioContext.decodeAudioData(
          audioBuffer.buffer.slice()
        );
        const source = audioContext.createBufferSource();
        source.buffer = audioData;
        source.connect(audioContext.destination);
        source.start();
      } catch (e) {
        // Not enough data yet, continue streaming
      }
    }
  }
}
```

## 📊 Performance Optimization

### Choosing Optimal Settings

**For Lowest Latency:**

```json
{
  "streaming_quality": "fast",
  "streaming_strategy": "word",
  "streaming_chunk_size": 80,
  "streaming_buffer_size": 1
}
```

**For Best Quality:**

```json
{
  "streaming_quality": "high",
  "streaming_strategy": "paragraph",
  "streaming_chunk_size": 350,
  "streaming_buffer_size": 5
}
```

**For Balanced Performance:**

```json
{
  "streaming_quality": "balanced",
  "streaming_strategy": "sentence",
  "streaming_chunk_size": 200,
  "streaming_buffer_size": 3
}
```

### Memory Optimization

The streaming implementation includes automatic memory management:

- Chunks are processed and freed immediately
- GPU memory is cleared periodically
- Temporary files are cleaned up automatically
- Progress tracking prevents memory leaks

## 🔄 Progress Monitoring

### Real-time Progress API

While streaming is active, you can monitor progress:

```bash
curl "http://localhost:4123/v1/status/progress"
```

**Response:**

```json
{
  "is_processing": true,
  "status": "generating_audio",
  "current_step": "Streaming audio for chunk 3/8",
  "current_chunk": 3,
  "total_chunks": 8,
  "progress_percentage": 37.5,
  "duration_seconds": 2.1,
  "estimated_completion": 1704067205.0,
  "text_preview": "This is the text being streamed..."
}
```

### Integration with Frontend

```typescript
// Monitor streaming progress
const monitorStreaming = async () => {
  const interval = setInterval(async () => {
    try {
      const response = await fetch('/v1/status/progress');
      const progress = await response.json();

      if (progress.is_processing) {
        updateProgressBar(progress.progress_percentage);
        updateStatus(progress.current_step);
      } else {
        clearInterval(interval);
        onStreamingComplete();
      }
    } catch (error) {
      console.error('Progress monitoring failed:', error);
    }
  }, 500);
};
```

## 🛠️ Troubleshooting

### Common Issues

**Streaming Stops Unexpectedly:**

- Check network stability
- Verify streaming headers are set correctly
- Ensure client supports chunked transfer encoding

**Audio Quality Issues:**

- Try larger `streaming_chunk_size`
- Use "sentence" or "paragraph" strategy
- Increase `streaming_quality` to "balanced" or "high"

**High Latency:**

- Reduce `streaming_chunk_size`
- Use "word" strategy for maximum responsiveness
- Set `streaming_quality` to "fast"

**Memory Issues:**

- Reduce `streaming_buffer_size`
- Use smaller `streaming_chunk_size`
- Monitor memory usage via `/memory` endpoint

### Debugging Commands

```bash
# Test streaming endpoint
curl -v -X POST http://localhost:4123/v1/audio/speech/stream \
  -H "Content-Type: application/json" \
  -d '{"input": "Test streaming"}' \
  --output debug_stream.wav

# Monitor memory during streaming
watch -n 1 'curl -s http://localhost:4123/memory | jq .memory_info'

# Check streaming progress
watch -n 0.5 'curl -s http://localhost:4123/v1/status/progress | jq .'
```

## 🔄 Comparison: Streaming vs Standard

### When to Use Streaming

**Use Streaming When:**

- Text length > 500 characters
- Building real-time applications
- Memory usage is a concern
- Users expect immediate audio feedback
- Implementing chat or interactive features

**Use Standard When:**

- Text length < 200 characters
- Need complete audio file before processing
- Working with simple integrations
- Bandwidth is limited
- Processing batch content

### Performance Comparison

| Aspect              | Standard Generation   | Streaming Generation  |
| ------------------- | --------------------- | --------------------- |
| **Initial Latency** | Full generation time  | ~1-2 seconds          |
| **Memory Usage**    | Peak during concat    | Constant low usage    |
| **User Experience** | Wait then play        | Progressive playback  |
| **Network Usage**   | Single large transfer | Multiple small chunks |
| **Complexity**      | Simple                | Moderate              |

## 🚀 Future Enhancements

Planned improvements to the streaming functionality:

- **Adaptive Chunking**: Automatically adjust chunk size based on content
- **Quality Adaptation**: Dynamic quality adjustment based on network conditions
- **Interruption Support**: Ability to stop streaming mid-generation
- **Buffer Prediction**: Intelligent buffering based on generation speed
- **Multi-voice Streaming**: Stream different voices for different speakers
- **WebSocket Support**: Real-time bidirectional streaming

## 📖 API Reference

For complete API documentation including all endpoints, parameters, and examples, visit:

- **Interactive Documentation**: http://localhost:4123/docs
- **Alternative Documentation**: http://localhost:4123/redoc
- **OpenAPI Schema**: http://localhost:4123/openapi.json

The streaming endpoints are fully documented with request/response schemas, parameter validation, and example payloads.
