# MLX Whisper Speech Transcriber

## Goal

Add an explicitly selected `mlx-whisper` speech-to-text backend for Apple Silicon speed while preserving the existing backend-generic `SpeechTranscriber` interface.

## Decision

Use `model_repo` as the public constructor field because MLX Whisper selects Hugging Face model repositories, not faster-whisper model-size names.

Example:

```python
from inspect_voice_eval import MlxWhisperSpeechTranscriber

transcriber = MlxWhisperSpeechTranscriber(
    model_repo="mlx-community/whisper-medium",
)
```

## Plan

1. Add `inspect_voice_eval/mlx_whisper_speech.py` with `MlxWhisperSpeechTranscriber`.
2. Implement the existing `SpeechTranscriber` protocol:
   - `backend_name` returns `"mlx-whisper"`;
   - `transcribe(audio: WaveData) -> str` runs MLX transcription off the async event loop;
   - audio is passed through `WaveData.to_wav_bytes()` using a short-lived `.wav` file, matching the existing faster-whisper backend pattern.
3. Add an optional dependency extra in `inspect-voice-eval/pyproject.toml`:
   - `mlx-whisper = ["mlx-whisper>=0.4,<1"]`
   - include it in the local all-in extra if one exists.
4. Export `MlxWhisperSpeechTranscriber` from `inspect_voice_eval.__init__`.
5. Add tests that avoid importing MLX unless installed:
   - constructor/backend name test;
   - optional backend import behavior;
   - mocked transcription path proving text is returned and audio reaches `mlx_whisper.transcribe`.
6. Update README backend examples to show explicit MLX selection.

## Non-goals

- Auto-select MLX on Apple Silicon.
- Remove or replace `FasterWhisperSpeechTranscriber`.
- Change `voice_turn` or `voice_conversation` APIs.
- Add live model downloads to normal tests.
