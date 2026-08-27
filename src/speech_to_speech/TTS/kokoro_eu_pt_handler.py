"""
Kokoro EU-PT TTS Handler

Wraps the `tts_eu_pt` package (https://huggingface.co/logus2k/kokoro_tts_eu_pt),
a European Portuguese (pt-PT) voice fine-tuned from Kokoro-82M with an
Apache-2.0 grapheme-to-phoneme front-end (TugaPhone), avoiding the GPLv3
espeak-ng dependency pulled in by the repo-native `kokoro` backend's
`KPipeline`/misaki G2P.

Single language (pt-PT), single voice. There is no lang_code/voice switching
here, unlike kokoro_handler.py, because the model only supports one language
and ships one voicepack.
"""

from __future__ import annotations

import logging
from threading import Event
from typing import Any, Iterator, Optional

import numpy as np
from rich.console import Console

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker

logger = logging.getLogger(__name__)
console = Console()


class KokoroEuPtTTSHandler(BaseHandler[TTSIn, TTSOut]):
    """
    Handles Text-to-Speech using the European Portuguese (pt-PT) Kokoro
    fine-tune from `tts_eu_pt`.
    """

    def setup(
        self,
        should_listen: Event,
        device: Optional[str] = None,
        model_path: Optional[str] = None,
        voicepack_path: Optional[str] = None,
        speed: float = 1.0,
        blocksize: int = 512,
        gen_kwargs: dict[str, Any] | None = None,
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
    ) -> None:
        """
        Initialize the tts_eu_pt model.

        Args:
            device: "cuda", "cpu", or None to auto-detect.
            model_path: Local path to tuga_kokoro.pth (defaults to package download).
            voicepack_path: Local path to tuga_voicepack.pt (defaults to package download).
            speed: Speech speed multiplier.
            blocksize: Audio chunk size for streaming.
            gen_kwargs: Unused, for pipeline compatibility.
        """
        self.should_listen = should_listen
        self.speed = speed
        self.blocksize = blocksize
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns

        try:
            from tts_eu_pt import TTS
        except ImportError as e:
            raise ImportError(
                "tts_eu_pt is required for the kokoro-eu-pt TTS backend. Install with: pip install tts_eu_pt"
            ) from e

        logger.info("Loading tts_eu_pt (Kokoro pt-PT) model")
        self.tts = TTS(
            device=device,
            model_path=model_path,
            voicepack_path=voicepack_path,
        )

        self.warmup()

    def warmup(self) -> None:
        """Warm up the model with a dummy inference."""
        logger.info(f"Warming up {self.__class__.__name__}")
        self.tts.say("Olá", speed=self.speed)
        logger.info(f"{self.__class__.__name__} warmed up")

    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        """
        Process text input and generate audio output.

        Yields:
            Audio chunks as numpy int16 arrays
        """
        speculative_turns = getattr(self, "speculative_turns", None)
        if isinstance(tts_input, EndOfResponse):
            if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
                tts_input.turn_id,
                tts_input.turn_revision,
            ):
                if tts_input.response_key is None:
                    return
                tts_input.cleanup_only = True
            yield AUDIO_RESPONSE_DONE
            return

        if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
            tts_input.turn_id,
            tts_input.turn_revision,
        ):
            logger.debug("Dropping stale TTS input for turn=%s rev=%s", tts_input.turn_id, tts_input.turn_revision)
            return
        if speculative_turns:
            speculative_turns.commit(tts_input.turn_id, tts_input.turn_revision)

        text = tts_input.text
        yield from self._process(text)

    def _process(self, llm_sentence: str) -> Iterator[np.ndarray]:
        """Synthesize with tts_eu_pt and resample/chunk for the pipeline."""
        from scipy.signal import resample_poly

        gen = self.cancel_scope.generation if self.cancel_scope else None

        console.print(f"[green]ASSISTANT: {llm_sentence}")

        audio = self.tts.say(llm_sentence, speed=self.speed)
        if audio is None or len(audio) == 0:
            return

        if not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)
        else:
            audio = audio.astype(np.float32)

        # tts_eu_pt outputs at 24kHz, resample to 16kHz for the pipeline.
        # 16000/24000 = 2/3, so up=2, down=3
        audio = resample_poly(audio, up=2, down=3)

        # Convert to int16 format
        audio = (audio * 32768).astype(np.int16)

        # Yield audio in fixed-size chunks
        for i in range(0, len(audio), self.blocksize):
            if gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(gen):
                logger.info("TTS generation cancelled (interruption)")
                return
            chunk = audio[i : i + self.blocksize]
            if len(chunk) < self.blocksize:
                chunk = np.pad(chunk, (0, self.blocksize - len(chunk)))
            yield chunk

    def on_session_end(self) -> None:
        logger.debug("Kokoro EU-PT TTS session state reset (stateless, nothing to do)")
