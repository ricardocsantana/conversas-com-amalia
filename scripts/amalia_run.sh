#!/usr/bin/env bash
# Launches "Conversas com Amália" — the speech-to-speech backend wired to
# teex/amalia (via a local Ollama server) and the kokoro_tts_eu_pt pt-PT voice.
#
# Run `bash scripts/amalia_setup.sh` first if you haven't already, and make
# sure the "amalia" conda env is active (or pass CONDA_ENV=name).
#
# Two modes:
#   MODE=local  (default) — mic/speaker loopback, talk right in this terminal.
#   MODE=serve             — WebSocket server for the browser demo (see demo/README.md).
#
# All defaults are overridable via environment variables, e.g.:
#   STT_DEVICE=cpu MODE=serve bash scripts/amalia_run.sh

set -euo pipefail

MODE="${MODE:-local}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434/v1}"
MODEL_NAME="${MODEL_NAME:-teex/amalia}"

# Whisper needs a literal device, not "auto". Default to mps on macOS
# (Apple Silicon), cpu elsewhere unless the caller overrides.
if [ -z "${STT_DEVICE:-}" ]; then
  if [ "$(uname -s)" = "Darwin" ]; then
    STT_DEVICE="mps"
  else
    STT_DEVICE="cpu"
  fi
fi

STT_MODEL="${STT_MODEL:-openai/whisper-large-v3-turbo}"
NUM_PIPELINES="${NUM_PIPELINES:-2}"
STREAM_BATCH_SENTENCES="${STREAM_BATCH_SENTENCES:-1}"

# The identity system prompt exactly as shipped in teex/amalia's Ollama
# Modelfile (`ollama show teex/amalia --modelfile`). Override with your own
# INIT_CHAT_PROMPT env var to change persona/tone.
DEFAULT_PROMPT="O teu nome é Amália e és um modelo de linguagem aberto desenvolvido em Portugal, com 9 mil milhões de parâmetros, treinado especialmente para o português europeu. O teu nome homenageia a fadista Amália Rodrigues. Foste apresentado publicamente a 1 de julho de 2026 e estás disponível em código aberto sob a licença Apache 2.0. Responde sempre na língua do utilizador, a menos que sejas instruído em contrário, e lembra-te que a tua língua principal é o português europeu. Quando te apresentares, diz que te chamas Amália, baseia-te apenas nesta informação e não inventes detalhes adicionais sobre a tua origem ou os teus criadores."
INIT_CHAT_PROMPT="${INIT_CHAT_PROMPT:-$DEFAULT_PROMPT}"

if ! curl -s -o /dev/null "http://localhost:11434/api/tags"; then
  echo "Ollama doesn't seem to be running on localhost:11434." >&2
  echo "Start it first (e.g. 'ollama serve' or the Ollama app), and make sure teex/amalia is pulled." >&2
  exit 1
fi

echo "==> Launching speech-to-speech ${MODE} (stt_device=${STT_DEVICE}, model=${MODEL_NAME})"

exec speech-to-speech "${MODE}" \
  --tts kokoro-eu-pt \
  --llm_backend chat-completions \
  --model_name "${MODEL_NAME}" \
  --responses_api_base_url "${OLLAMA_URL}" \
  --responses_api_api_key ollama \
  --stt whisper --language pt \
  --stt_model_name "${STT_MODEL}" \
  --stt_device "${STT_DEVICE}" \
  --log_transcripts \
  --num_pipelines "${NUM_PIPELINES}" \
  --stream_batch_sentences "${STREAM_BATCH_SENTENCES}" \
  --init_chat_prompt "${INIT_CHAT_PROMPT}"
