# TTS Backend Feature Matrix

This matrix is a decision aid before benchmarking. It separates features we care about for black-box voice evals: latency, intelligibility, expressive tone, multiple voices, reference-voice support, local/hosted deployment, and Mac suitability.

| Backend | Local or hosted | Expected latency | Naturalness | Tone / expressiveness | Multiple voices | Reference voice / cloning | Mac Apple Silicon fit | Streaming | Main risk | Best use in our eval |
|---|---|---:|---:|---|---|---|---|---|---|---|
| Chatterbox | Local | Slow on our Mac | High | Yes, expressiveness / CFG controls | Limited / unclear | Yes | Poor from observed CPU/MPS timings | Unclear | Too slow for live turns | High-quality local prompts when latency is not critical |
| Chatterbox Turbo | Local | Unknown, likely faster than Chatterbox | Likely high | Likely yes | Limited / unclear | Likely yes | Unknown | Unknown | May still be slow on Mac | First same-family speed experiment |
| Kokoro | Local | Fast expected | Medium-high | Limited, voice/style dependent | Yes | No or limited | Good expected | Possible through wrappers | Less expressive than Chatterbox | Best likely local speed/naturalness candidate |
| Piper | Local | Very fast | Medium | Low | Yes | No | Good | Usually whole utterance | Voice can sound synthetic | Fast intelligible baseline for eval prompts |
| MeloTTS | Local | Fast expected | Medium-high | Some style/language controls | Yes | Limited / unclear | Okay expected | Unclear | Package maturity / setup | Middle ground between Piper speed and naturalness |
| F5-TTS / E2-TTS | Local | Medium to slow expected | High | High via prompt/reference style | Depends on checkpoints | Yes | Unclear, likely not ideal | Unclear | Setup complexity and speed | Expressive local voice cloning if speed is acceptable |
| Fish Speech / CosyVoice | Local | Medium to slow expected | High | High | Yes | Yes | Unclear, likely heavy | Some variants may stream | Model size/setup/license | High-quality local speech if hardware is acceptable |
| ElevenLabs | Hosted | Fast perceived | Very high | High | Yes | Yes | API, not hardware-bound | Yes | Cost/privacy/vendor lock-in | Best hosted expressiveness if external service is acceptable |
| OpenAI TTS | Hosted | Fast perceived | High | Some voice/style control | Yes | No or limited | API, not hardware-bound | Some APIs support streaming | Cost/privacy/vendor lock-in | Simple hosted baseline with good quality |
| Cartesia | Hosted | Very fast perceived | High | High | Yes | Voice features vary | API, not hardware-bound | Yes | Cost/privacy/vendor lock-in | Low-latency hosted candidate |
| Deepgram Aura | Hosted | Very fast perceived | High | Some style/voice selection | Yes | No or limited | API, not hardware-bound | Yes | Cost/privacy/vendor lock-in | Low-latency hosted speech for interactive evals |

## Features that matter most for us

1. **Latency per prompt**: must be low enough that live evals do not stall.
2. **Intelligibility**: target chatbot must reliably hear the prompt.
3. **Naturalness**: useful, but less important than intelligibility unless target behavior changes with robotic voices.
4. **Tone control**: important only if we test emotional/prosodic behavior.
5. **Multiple voices**: useful for robustness testing across speakers.
6. **Reference voice / cloning**: nice for realism, not required for basic black-box evals.
7. **Local vs hosted**: local preserves privacy/reproducibility; hosted may win latency and quality.

## First benchmark shortlist

1. Chatterbox Turbo — lowest-friction comparison against current backend.
2. Kokoro — likely best local speed/naturalness tradeoff.
3. Piper — fastest local intelligibility baseline.
4. One hosted option — only if local backends remain too slow.
