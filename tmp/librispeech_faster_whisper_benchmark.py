from __future__ import annotations

import re
import time
from dataclasses import dataclass

try:
    from datasets import load_dataset
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Install the dataset loader first: uv pip install 'datasets[audio]'"
    ) from exc

from faster_whisper import WhisperModel

DATASET_NAME = "openslr/librispeech_asr"
DATASET_VARIANT = "clean"
DATASET_SPLIT = "test"
SAMPLE_COUNT = 3
MODEL_SIZES = ("tiny", "base", "small", "medium", "large-v3")
DEVICE = "cpu"


@dataclass(frozen=True)
class SpeechExample:
    """Reference transcript and audio samples for one benchmark clip."""

    text: str
    samples: object
    duration_seconds: float


@dataclass(frozen=True)
class ClipResult:
    """One transcription result."""

    model_size: str
    reference_text: str
    predicted_text: str
    reference_words: int
    word_errors: int
    latency_seconds: float
    duration_seconds: float


def normalized_words(text: str) -> list[str]:
    """Return lowercase words with punctuation removed."""
    cleaned = re.sub(r"[^a-z0-9']+", " ", text.lower())
    return [word for word in cleaned.split() if word]


def word_errors(reference_words: list[str], predicted_words: list[str]) -> int:
    """Return Levenshtein word errors."""
    previous_row = list(range(len(predicted_words) + 1))
    for reference_index, reference_word in enumerate(reference_words, start=1):
        current_row = [reference_index]
        for predicted_index, predicted_word in enumerate(predicted_words, start=1):
            substitution_cost = 0 if reference_word == predicted_word else 1
            current_row.append(
                min(
                    previous_row[predicted_index] + 1,
                    current_row[predicted_index - 1] + 1,
                    previous_row[predicted_index - 1] + substitution_cost,
                )
            )
        previous_row = current_row
    return previous_row[-1]


def load_examples() -> list[SpeechExample]:
    """Load a few LibriSpeech examples with reference transcripts."""
    stream = load_dataset(
        DATASET_NAME,
        DATASET_VARIANT,
        split=DATASET_SPLIT,
        streaming=True,
    )
    examples: list[SpeechExample] = []
    for row in stream:
        audio = row["audio"]
        samples = audio["array"]
        sampling_rate = audio["sampling_rate"]
        examples.append(
            SpeechExample(
                text=row["text"],
                samples=samples,
                duration_seconds=len(samples) / sampling_rate,
            )
        )
        if len(examples) >= SAMPLE_COUNT:
            break
    return examples


def transcribe_examples(
    model_size: str, examples: list[SpeechExample]
) -> list[ClipResult]:
    """Run one faster-whisper model over examples."""
    load_started = time.perf_counter()
    model = WhisperModel(model_size, device=DEVICE)
    print(f"\nmodel: {model_size}")
    print(f"model_load_seconds: {time.perf_counter() - load_started:.3f}")

    results: list[ClipResult] = []
    for example_index, example in enumerate(examples, start=1):
        started = time.perf_counter()
        segments, _ = model.transcribe(example.samples)
        predicted_text = " ".join(segment.text.strip() for segment in segments).strip()
        latency_seconds = time.perf_counter() - started
        reference_words = normalized_words(example.text)
        predicted_words = normalized_words(predicted_text)
        errors = word_errors(reference_words, predicted_words)
        result = ClipResult(
            model_size=model_size,
            reference_text=example.text,
            predicted_text=predicted_text,
            reference_words=len(reference_words),
            word_errors=errors,
            latency_seconds=latency_seconds,
            duration_seconds=example.duration_seconds,
        )
        results.append(result)
        print(f"\nclip: {example_index}")
        print(f"duration_seconds: {result.duration_seconds:.3f}")
        print(f"reference: {result.reference_text}")
        print(f"prediction: {result.predicted_text}")
        print(f"word_errors: {result.word_errors}/{result.reference_words}")
        print(f"latency_seconds: {result.latency_seconds:.3f}")
    return results


def print_summary(model_size: str, results: list[ClipResult]) -> None:
    """Print total word errors and latency for one model."""
    total_errors = sum(result.word_errors for result in results)
    total_words = sum(result.reference_words for result in results)
    total_latency = sum(result.latency_seconds for result in results)
    total_duration = sum(result.duration_seconds for result in results)
    print(f"\nsummary model: {model_size}")
    print(f"word_errors: {total_errors}/{total_words}")
    print(f"latency_seconds: {total_latency:.3f}")
    print(f"audio_seconds: {total_duration:.3f}")
    print(f"latency_per_audio_second: {total_latency / total_duration:.3f}")


def main() -> None:
    """Run the small faster-whisper LibriSpeech benchmark."""
    examples = load_examples()
    print(f"loaded_examples: {len(examples)}")
    for model_size in MODEL_SIZES:
        results = transcribe_examples(model_size, examples)
        print_summary(model_size, results)


if __name__ == "__main__":
    main()
