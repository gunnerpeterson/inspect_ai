# Kokoro and Piper Speech Synthesizers

## Goal

Add explicitly selected local TTS backends for faster prompt audio generation:

- `KokoroSpeechSynthesizer` for better local naturalness/speed balance.
- `PiperSpeechSynthesizer` for fastest local intelligible prompts.

Both must implement the existing `SpeechSynthesizer` protocol and preserve the current `voice_turn` / `voice_conversation` APIs.

## Evidence

Local speed audit for the prompt `Hello. Please introduce yourself in one short sentence.`:

| Backend | Latency | Audio duration | Latency per audio second |
|---|---:|---:|---:|
| Piper | 1.071s | 3.390s | 0.316 |
| Kokoro | 6.587s | 3.725s | 1.768 |
| Chatterbox Turbo | 16.251s | 3.360s | 4.837 |
| Chatterbox | 28.667s | 3.320s | 8.635 |

## Public API

```python
from inspect_voice_eval import KokoroSpeechSynthesizer, PiperSpeechSynthesizer

kokoro = KokoroSpeechSynthesizer(voice_id="af_heart")
piper = PiperSpeechSynthesizer(voice_id="en_US-lessac-medium")
```

Both are explicitly selected by the caller. No automatic backend choice.

## Voice controls

### Kokoro

Supported controls:

- `voice_id`
- `speaking_rate`

Defaults:

- `voice_id="af_heart"`
- `language_code="a"`
- `repo_id="hexgrad/Kokoro-82M"`

Kokoro supports multiple voices. Tone is controlled mainly by voice choice, not direct emotional controls.

### Piper

Supported controls:

- `voice_id`
- `speaking_rate`

Defaults:

- `voice_id="en_US-lessac-medium"`

Piper supports multiple voice files. Tone control is weak; speed is the main useful control.

## Plan

1. Add `kokoro_speech.py` with `KokoroSpeechSynthesizer`.
   - Lazily import `kokoro`.
   - Keep the Kokoro pipeline cached on the synthesizer instance.
   - Convert float samples to `WaveData`.
   - Raise `UnsupportedVoiceControlError` through the existing rejection helper for unsupported controls.
2. Add `piper_speech.py` with `PiperSpeechSynthesizer`.
   - Lazily import `piper.download_voices`.
   - Download/cache the selected voice under a caller-specified local directory, defaulting under the user cache directory.
   - Invoke the `piper` CLI from the active Python environment or PATH.
   - Read generated WAV with `soundfile` and convert to `WaveData`.
3. Add optional extras in `inspect-voice-eval/pyproject.toml`:
   - `kokoro = ["kokoro>=0.9,<1", "soundfile>=0.12,<1"]`
   - `piper = ["piper-tts>=1.8,<2", "soundfile>=0.12,<1"]`
   - include both in `local`.
4. Export both synthesizers from `inspect_voice_eval.__init__`.
5. Add tests with mocked optional packages/CLI behavior:
   - backend names and supported controls;
   - constructor does not import optional packages;
   - unsupported controls fail;
   - Kokoro converts generated samples to `WaveData`;
   - Piper downloads voice, invokes CLI, and returns `WaveData`.
6. Update README with explicit Kokoro and Piper examples.

## Non-goals

- Auto-select Kokoro or Piper.
- Remove Chatterbox.
- Add hosted TTS.
- Add direct emotional tone controls.
- Commit generated audio artifacts.
