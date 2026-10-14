# MLX Whisper Smoke Benchmark

## Goal

Test whether Apple Silicon GPU acceleration through `mlx-whisper` can transcribe the same small LibriSpeech set faster than `faster-whisper medium` while preserving word accuracy.

## Plan

1. Install `mlx-whisper` into the local virtual environment.
2. Run the same 3 LibriSpeech examples used by `tmp/librispeech_faster_whisper_benchmark.py`.
3. Measure word errors and latency seconds for an MLX Whisper model close to the chosen `medium` quality tier.
4. Compare the result against the existing `faster-whisper medium` result:
   - word errors: `0 / 62`
   - latency: `18.203s`
   - audio duration: `22.755s`

## Non-goals

- Change the voice extension API.
- Replace `faster-whisper` yet.
- Add committed audio fixtures.
- Add a full benchmark suite.

## Result

The cached `mlx-community/whisper-medium` run completed on the same 3 streamed examples loaded by `tmp/librispeech_faster_whisper_benchmark.py`.

| Backend | Model | Word errors | Latency | Audio duration | Latency per audio second |
|---|---|---:|---:|---:|---:|
| faster-whisper | medium CPU | 0 / 62 | 18.203s | 22.755s | 0.800 |
| mlx-whisper | mlx-community/whisper-medium | 0 / 62 | 2.707s | 22.755s | 0.119 |

`mlx-whisper` was about 6.7x faster than the `faster-whisper medium` CPU baseline on this small cached run, with the same word error count.
