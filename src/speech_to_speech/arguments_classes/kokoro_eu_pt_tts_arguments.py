from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KokoroEuPtTTSHandlerArguments:
    kokoro_eu_pt_device: Optional[str] = field(
        default=None,
        metadata={"help": "Device to run tts_eu_pt on: 'cuda', 'cpu', or None to auto-detect. Default is None."},
    )
    kokoro_eu_pt_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Local path to tuga_kokoro.pth. Defaults to the package's bundled download."},
    )
    kokoro_eu_pt_voicepack_path: Optional[str] = field(
        default=None,
        metadata={"help": "Local path to tuga_voicepack.pt. Defaults to the package's bundled download."},
    )
    kokoro_eu_pt_speed: float = field(
        default=1.0,
        metadata={"help": "Speech speed multiplier. Values > 1.0 speed up, < 1.0 slow down. Default is 1.0."},
    )
    kokoro_eu_pt_blocksize: int = field(
        default=512,
        metadata={"help": "The audio chunk size in samples for streaming output. Default is 512."},
    )
