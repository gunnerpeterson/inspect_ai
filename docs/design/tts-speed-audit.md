# TTS Speed Audit

## Goal

Benchmark candidate TTS backends for prompt generation speed and retain listenable MP3 samples for subjective quality review.

## Candidate backends

Only local candidates are in scope for this first audit:

- Chatterbox
- Chatterbox Turbo
- Kokoro
- Piper
- MeloTTS
- F5-TTS / E2-TTS
- Fish Speech / CosyVoice

## What to measure

Use one fixed prompt first:

> Hello. Please introduce yourself in one short sentence.

For each backend that can run in the current environment, record:

- synthesis latency seconds;
- generated audio duration seconds;
- latency per generated audio second;
- output file path;
- error message for unavailable backends.

## Output artifacts

Write local, ignored artifacts under `tmp/tts_speed_audit/`:

- one MP3 per successful backend;
- one machine-readable JSON result file;
- one Markdown summary table.

These artifacts are for local listening and should not be committed.

## Implementation plan

1. Create a local audit script at `tmp/tts_speed_audit.py`.
2. Implement each backend as an optional probe so missing local packages do not stop the whole audit.
3. Skip hosted TTS providers entirely in this first audit.
4. Convert generated audio to MP3 with available local tooling.
5. Print and save a summary sorted by latency.

## Non-goals

- Add TTS backends to the Inspect extension during this audit.
- Commit generated MP3 files.
- Require every candidate dependency to be installed before the script can run.
- Call hosted TTS APIs.
- Judge subjective voice quality automatically.

## Result

The local audit script prints and saves a summary sorted by latency. Successful runs write MP3 samples, WAV files, `results.json`, and `summary.md` under `tmp/tts_speed_audit/`.
