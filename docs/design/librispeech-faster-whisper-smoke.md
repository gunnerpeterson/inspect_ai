# LibriSpeech faster-whisper Smoke Benchmark

## Goal

Add a local-only script that pulls a small number of LibriSpeech examples and compares faster-whisper model sizes by:

- word errors against the LibriSpeech transcript;
- latency in seconds.

## Scope

Use a tiny sample count for quick local testing. The script is a local smoke benchmark, not a committed benchmark artifact.

## Plan

Create `tmp/librispeech_faster_whisper_benchmark.py`.

The script should:

1. load a few LibriSpeech clips from Hugging Face `librispeech_asr`;
2. run faster-whisper model sizes selected in the script;
3. print reference text, prediction text, word errors, reference word count, latency seconds, and total summary per model;
4. avoid extra scoring dependencies by using a small local word edit-distance function.

## Non-goals

- full ASR benchmark suite;
- hosted ASR comparison;
- committed audio fixtures;
- disk cache design;
- voice extension API changes.
