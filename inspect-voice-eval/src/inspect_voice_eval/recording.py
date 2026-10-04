from __future__ import annotations

from dataclasses import dataclass, field

from .audio_endpoint import RecordingResult, VoiceTurnTimeoutError
from .speech_detection import VoiceActivityDetector
from .wave_data import WaveData


@dataclass(frozen=True)
class WaitingForSpeech:
    """Recording phase before any speech frame has arrived."""


@dataclass(frozen=True)
class HearingSpeech:
    """Recording phase after speech has started."""

    speech_started_ms: int
    latest_speech_end_ms: int
    silence_after_speech_ms: int = 0


RecordingPhase = WaitingForSpeech | HearingSpeech


@dataclass(frozen=True)
class ContinueRecording:
    """Frame ingestion outcome when more audio is needed."""


@dataclass(frozen=True)
class SpeechRecorded:
    """Frame ingestion outcome when speech has completed."""

    recording: RecordingResult


RecordingProgress = ContinueRecording | SpeechRecorded


@dataclass
class SpeechStopState:
    """Incremental speech stop detector for recorded audio frames."""

    detector: VoiceActivityDetector
    end_silence_ms: int
    min_speech_ms: int
    frames: list[WaveData] = field(default_factory=list)
    # The phase classes add a little ceremony, but they make missing speech
    # timestamps unrepresentable once speech has started.
    phase: RecordingPhase = field(default_factory=WaitingForSpeech)

    def accept(self, frame_start_ms: int, frame: WaveData) -> RecordingProgress:
        """Accept one audio frame and return the named recording outcome."""
        self.frames.append(frame)
        frame_end_ms = frame_start_ms + frame.duration_ms
        decision = self.detector.detect(frame)
        if decision.state == "speech":
            if isinstance(self.phase, WaitingForSpeech):
                self.phase = HearingSpeech(
                    speech_started_ms=frame_start_ms,
                    latest_speech_end_ms=frame_end_ms,
                )
                return ContinueRecording()
            self.phase = HearingSpeech(
                speech_started_ms=self.phase.speech_started_ms,
                latest_speech_end_ms=frame_end_ms,
            )
            return ContinueRecording()

        if isinstance(self.phase, WaitingForSpeech):
            return ContinueRecording()

        phase = HearingSpeech(
            speech_started_ms=self.phase.speech_started_ms,
            latest_speech_end_ms=self.phase.latest_speech_end_ms,
            silence_after_speech_ms=self.phase.silence_after_speech_ms
            + frame.duration_ms,
        )
        self.phase = phase
        speech_duration_ms = phase.latest_speech_end_ms - phase.speech_started_ms
        if (
            speech_duration_ms >= self.min_speech_ms
            and phase.silence_after_speech_ms >= self.end_silence_ms
        ):
            recorded_audio = WaveData.concatenate(self.frames).slice(
                phase.speech_started_ms, phase.latest_speech_end_ms
            )
            return SpeechRecorded(
                RecordingResult(
                    audio=recorded_audio,
                    speech_started_ms=phase.speech_started_ms,
                    speech_completed_ms=phase.latest_speech_end_ms,
                )
            )
        return ContinueRecording()

    def timeout(self) -> VoiceTurnTimeoutError:
        """Return the timeout error matching the current speech phase."""
        if isinstance(self.phase, HearingSpeech):
            return VoiceTurnTimeoutError("Speech did not end before recording timeout")
        return VoiceTurnTimeoutError("No speech detected before recording timeout")


def recording_from_audio(
    audio: WaveData,
    detector: VoiceActivityDetector,
    *,
    frame_duration_ms: int,
    end_silence_ms: int,
    min_speech_ms: int,
    max_recording_ms: int,
) -> RecordingResult:
    """Segment recorded audio with a VAD stopping rule."""
    stop_state = SpeechStopState(
        detector=detector,
        end_silence_ms=end_silence_ms,
        min_speech_ms=min_speech_ms,
    )

    for frame_start_ms, frame in audio.frames(frame_duration_ms):
        frame_end_ms = frame_start_ms + frame.duration_ms
        if frame_end_ms > max_recording_ms:
            break
        progress = stop_state.accept(frame_start_ms, frame)
        if isinstance(progress, SpeechRecorded):
            return progress.recording

    raise stop_state.timeout()
