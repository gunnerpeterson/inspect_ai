from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.log import read_eval_log

from inspect_voice_eval import (
    FakeAudioEndpoint,
    FakeSpeechSynthesizer,
    FakeSpeechTranscriber,
    VoiceTurnTimingSettings,
    WaveData,
    voice_turn,
)
from inspect_voice_eval.log_attachment_refs import attachment_ref_for_text


def test_log_artifact_refs_match_written_eval_log_attachments(tmp_path) -> None:
    prompt_audio = WaveData.tone(180)
    response_audio = WaveData.concatenate(
        [WaveData.silence(90), WaveData.tone(300), WaveData.silence(300)]
    )
    recorded_audio = response_audio.slice(90, 390)
    task = Task(
        dataset=[Sample(input="Say hello", target="hello")],
        solver=voice_turn(
            synthesizer=FakeSpeechSynthesizer(prompt_audio),
            audio_endpoint=FakeAudioEndpoint(response_audio),
            transcriber=FakeSpeechTranscriber.for_audio(recorded_audio, "hello"),
            timing=VoiceTurnTimingSettings(max_recording_ms=2_000),
        ),
    )

    log = eval(task, model="mockllm/model", log_dir=str(tmp_path), display="none")[0]
    reread = read_eval_log(log.location)

    assert reread.samples is not None
    sample = reread.samples[0]
    record = sample.store["voice_turn"]
    prompt_ref = record["artifacts"]["prompt_audio_ref"]
    response_ref = record["artifacts"]["response_audio_ref"]

    assert prompt_ref == attachment_ref_for_text(prompt_audio.to_wav_data_uri())
    assert response_ref == attachment_ref_for_text(recorded_audio.to_wav_data_uri())
    assert (
        sample.attachments[prompt_ref.removeprefix("attachment://")]
        == prompt_audio.to_wav_data_uri()
    )
    assert (
        sample.attachments[response_ref.removeprefix("attachment://")]
        == recorded_audio.to_wav_data_uri()
    )
