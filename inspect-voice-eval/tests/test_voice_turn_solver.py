import anyio
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match

from inspect_voice_eval import (
    FakeAudioEndpoint,
    FakeSpeechSynthesizer,
    FakeSpeechTranscriber,
    VoiceRequest,
    VoiceTurnTimingSettings,
    WaveData,
    voice_turn,
)
from inspect_voice_eval.log_attachment_refs import attachment_ref_for_text
from inspect_voice_eval.speech_synthesis import reject_unsupported_controls


def response_audio() -> WaveData:
    return WaveData.concatenate(
        [WaveData.silence(90), WaveData.tone(300), WaveData.silence(300)]
    )


def recorded_response_audio() -> WaveData:
    return response_audio().slice(90, 390)


def test_voice_turn_solver_scores_transcript_and_records_artifacts(tmp_path) -> None:
    prompt_audio = WaveData.tone(180)
    endpoint = FakeAudioEndpoint(response_audio())
    task = Task(
        dataset=[Sample(input="Say hello", target="hello")],
        solver=voice_turn(
            synthesizer=FakeSpeechSynthesizer(prompt_audio),
            audio_endpoint=endpoint,
            transcriber=FakeSpeechTranscriber.for_audio(
                recorded_response_audio(), "hello"
            ),
            timing=VoiceTurnTimingSettings(max_recording_ms=2_000),
        ),
        scorer=match(location="exact"),
    )

    log = eval(task, model="mockllm/model", log_dir=str(tmp_path), display="none")[0]

    assert log.samples is not None
    sample = log.samples[0]
    assert sample.output.completion == "hello"
    assert sample.scores is not None
    assert sample.scores["match"].value == "C"
    assert len(endpoint.played_audio) == 1

    record = sample.store["voice_turn"]
    prompt_ref = record["artifacts"]["prompt_audio_ref"]
    response_ref = record["artifacts"]["response_audio_ref"]
    assert prompt_ref == attachment_ref_for_text(prompt_audio.to_wav_data_uri())
    assert response_ref == attachment_ref_for_text(
        sample.attachments[response_ref.removeprefix("attachment://")]
    )
    assert prompt_ref.removeprefix("attachment://") in sample.attachments
    assert response_ref.removeprefix("attachment://") in sample.attachments
    assert record["backends"] == {
        "tts": "fake-tts",
        "audio": "fake-audio",
        "vad": "energy-threshold",
        "transcriber": "fake-transcriber",
    }
    assert record["audio"]["sample_rate_hz"] == 16_000
    assert (
        record["timing"]["speech_started_ms"]
        >= record["timing"]["recording_started_ms"]
    )


def test_voice_turn_solver_rejects_unsupported_voice_controls(tmp_path) -> None:
    task = Task(
        dataset=[Sample(input="Say hello", target="hello")],
        solver=voice_turn(
            synthesizer=FakeSpeechSynthesizer(),
            audio_endpoint=FakeAudioEndpoint(response_audio()),
            transcriber=FakeSpeechTranscriber.for_audio(
                recorded_response_audio(), "hello"
            ),
            voice=VoiceRequest(voice_id="speaker-a"),
        ),
        scorer=match(),
    )

    log = eval(task, model="mockllm/model", log_dir=str(tmp_path), display="none")[0]

    assert log.status == "error"
    assert log.samples is not None
    assert log.samples[0].error is not None
    assert "does not support requested voice controls" in log.samples[0].error.message


def test_voice_turn_solver_accepts_backend_factories(tmp_path) -> None:
    task = Task(
        dataset=[Sample(input="Say hello", target="hello")],
        solver=voice_turn(
            synthesizer=lambda: FakeSpeechSynthesizer(WaveData.tone(180)),
            audio_endpoint=lambda: FakeAudioEndpoint(response_audio()),
            transcriber=lambda: FakeSpeechTranscriber.for_audio(
                recorded_response_audio(), "hello"
            ),
            timing=VoiceTurnTimingSettings(max_recording_ms=2_000),
        ),
        scorer=match(location="exact"),
    )

    log = eval(task, model="mockllm/model", log_dir=str(tmp_path), display="none")[0]

    assert log.samples is not None
    assert log.samples[0].output.completion == "hello"


class SlowSpeechSynthesizer:
    """Synthesizer that exceeds the test timeout."""

    @property
    def backend_name(self) -> str:
        """Stable backend name for logs."""
        return "slow-tts"

    @property
    def supported_controls(self) -> set[str]:
        """Voice controls this backend can honor."""
        return set()

    async def synthesize(self, text: str, voice: VoiceRequest) -> WaveData:
        """Sleep longer than the configured timeout."""
        reject_unsupported_controls(self.backend_name, self.supported_controls, voice)
        await anyio.sleep(0.2)
        return WaveData.tone(100)


def test_voice_turn_solver_bounds_synthesis_wait(tmp_path) -> None:
    task = Task(
        dataset=[Sample(input="Say hello", target="hello")],
        solver=voice_turn(
            synthesizer=SlowSpeechSynthesizer(),
            audio_endpoint=FakeAudioEndpoint(response_audio()),
            transcriber=FakeSpeechTranscriber.for_audio(
                recorded_response_audio(), "hello"
            ),
            timing=VoiceTurnTimingSettings(
                synthesis_timeout_ms=10,
                max_recording_ms=2_000,
            ),
        ),
        scorer=match(),
    )

    log = eval(task, model="mockllm/model", log_dir=str(tmp_path), display="none")[0]

    assert log.status == "error"
    assert log.samples is not None
    assert log.samples[0].error is not None


def test_voice_turn_solver_surfaces_transcription_failure(tmp_path) -> None:
    task = Task(
        dataset=[Sample(input="Say hello", target="hello")],
        solver=voice_turn(
            synthesizer=FakeSpeechSynthesizer(),
            audio_endpoint=FakeAudioEndpoint(response_audio()),
            transcriber=FakeSpeechTranscriber(fail=True),
        ),
        scorer=match(),
    )

    log = eval(task, model="mockllm/model", log_dir=str(tmp_path), display="none")[0]

    assert log.status == "error"
    assert log.samples is not None
    assert log.samples[0].error is not None
    assert "fake transcription failure" in log.samples[0].error.message
