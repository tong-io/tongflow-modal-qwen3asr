"""
Qwen3-ASR on Modal. TypeScript handlers: src/handlers/modal/qwen3-asr.ts
(features: transcribe → qwen3-asr, transcribe_timestamp → qwen3-asr-timestamp).

Callers (openflow ``qwen3-asr.ts``) send ``audio_bytes`` + ``filename`` after fetching
via ``fetchModalAssetBytes`` (supports ``/api/uploads/...`` and HTTPS). Video files
are demuxed to 16 kHz mono WAV with ffmpeg before ASR.

Deploy: modal deploy modal/gpu/qwen3asr.py
Models:  modal run modal/gpu/qwen3asr.py::download
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional, cast

import modal
from tongflow import deploy




_cfg: dict[str, Any] = {}


def _hf_repo_pairs() -> list[tuple[str, str]]:
    _hf = _cfg.get("hf") if isinstance(_cfg.get("hf"), dict) else {}
    repos = _hf.get("repos")
    if isinstance(repos, list) and len(repos) >= 2:
        out: list[tuple[str, str]] = []
        for r in repos:
            if isinstance(r, dict) and r.get("repoId"):
                out.append((str(r["repoId"]), str(r.get("revision") or "")))
        if len(out) >= 2:
            return out[:2]
    return [
        ("Qwen/Qwen3-ASR-1.7B", ""),
        ("Qwen/Qwen3-ForcedAligner-0.6B", ""),
    ]


_p = _hf_repo_pairs()
ASR_REPO_ID, _ = _p[0]
ALIGNER_REPO_ID, _ = _p[1]
ASR_MODEL_DIR = f"/models/{ASR_REPO_ID}"
ALIGNER_MODEL_DIR = f"/models/{ALIGNER_REPO_ID}"

_volume_name = str(_cfg.get("volumeName") or "models")
volume = modal.Volume.from_name(_volume_name, create_if_missing=True)

from tongflow.models.transcribe import TranscribeInput, TranscribeOutput
from tongflow.models.transcribe_timestamp import (
    TranscribeTimestampInput,
    TranscribeTimestampOutput,
    TranscribeTimestampOutputRootTimeStampsItem,
)
from tongflow.node_slots import NodeSlots
from tongflow.protocol import prompt_media_to_bytes
from tongflow.slots import node_slot

# ── app ──────────────────────────────────────────────────────────────────────

app = modal.App(Path(__file__).resolve().parent.name)

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
    .apt_install("ffmpeg", "sox", "libsox-dev")
    .pip_install(
        "tongflow==0.2.21", "fastapi[standard]",
        "qwen-asr==0.0.6",
        "transformers==4.57.6",
        "accelerate==1.12.0",
        "soundfile==0.13.1",
        "librosa==0.10.2.post1",
        "torchaudio",
        "huggingface_hub>=0.34.0,<1.0",
        "flash-attn>=2.5.0",
    )
)

with image.imports():
    import torch
    from qwen_asr import Qwen3ASRModel

# ── video → audio (ffmpeg), runs in the same container as ASR ───────────────

_VIDEO_EXTS = frozenset(
    {
        ".mp4",
        ".webm",
        ".mov",
        ".mkv",
        ".avi",
        ".m4v",
        ".mpeg",
        ".mpg",
        ".flv",
        ".wmv",
        ".3gp",
    }
)


def _ffmpeg_extract_wav(video_path: Path, wav_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(wav_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg extract audio failed: {proc.stderr or proc.stdout}"
        )


@contextmanager
def _asr_audio_input_from_bytes(
    audio_bytes: bytes, filename: str
) -> Generator[str, None, None]:
    """
    Write ``audio_bytes`` to a temp file; if extension is a known video type,
    ffmpeg-extract WAV, else pass the file path to Qwen ASR (audio formats).
    """
    name = filename.strip() or "media.bin"
    suffix = Path(name).suffix.lower() or ".bin"
    with tempfile.TemporaryDirectory(prefix="qwen3asr_") as td:
        tdir = Path(td)
        media_path = tdir / f"input{suffix}"
        media_path.write_bytes(audio_bytes)
        if suffix in _VIDEO_EXTS:
            wav = tdir / "for_asr.wav"
            _ffmpeg_extract_wav(media_path, wav)
            yield str(wav)
        else:
            yield str(media_path)


def _align_result_to_list(result):
    if result is None:
        return None
    return [
        {
            "start": float(it.start_time),
            "end": float(it.end_time),
            "text": it.text,
        }
        for it in result.items
    ]


@deploy
@app.cls(
    scaledown_window=2,
    image=image,
    gpu="L4",
    volumes={"/models": volume},
)
class Transcribe:
    """Qwen3-ASR-1.7B only (no forced aligner). Lower VRAM than `TranscribeWithTimestamps`."""

    @modal.enter()
    def load(self):
        self.asr = Qwen3ASRModel.from_pretrained(
            ASR_MODEL_DIR,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            max_inference_batch_size=32,
            max_new_tokens=512,
        )

    def _transcribe_from_bytes(
        self,
        audio_bytes: bytes,
        filename: str = "media.bin",
        context: str = "",
        language: Optional[str] = None,
        max_new_tokens: int = 512,
    ) -> dict:
        self.asr.max_new_tokens = max_new_tokens
        lang = language if (language and str(language).strip()) else None
        with _asr_audio_input_from_bytes(audio_bytes, filename) as asr_src:
            results = self.asr.transcribe(
                audio=asr_src,
                context=context,
                language=lang,
                return_time_stamps=False,
            )
        r = results[0]
        return {"language": r.language, "text": r.text}

    @modal.method()
    def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "media.bin",
        context: str = "",
        language: Optional[str] = None,
        max_new_tokens: int = 512,
    ) -> dict:
        """
        Transcribe from raw file bytes (same pattern as ``flux2_klein9b`` ``image: bytes``).
        ``filename`` suffix selects video (ffmpeg) vs audio path.
        """
        return self._transcribe_from_bytes(
            audio_bytes=audio_bytes,
            filename=filename,
            context=context,
            language=language,
            max_new_tokens=max_new_tokens,
        )

    @modal.method()
    @node_slot(NodeSlots.TRANSCRIBE)
    def transcribe_openflow(self, input: TranscribeInput) -> TranscribeOutput:
        if input.audio is None:
            return TranscribeOutput(success=False, error="Missing `audio` Asset")
        out = self._transcribe_from_bytes(
            audio_bytes=prompt_media_to_bytes(input.audio),
            filename=input.audio.filename or "media.bin",
            context=input.context or "",
            language=input.language or None,
            max_new_tokens=int(input.max_new_tokens)
            if input.max_new_tokens is not None
            else 512,
        )
        return TranscribeOutput(success=True, text=str(out.get("text") or ""))

    @modal.fastapi_endpoint(method="GET", label=f"{Path(__file__).resolve().parent.name}-transcribe-serve")
    def serve(self, taskId: str = "", token: str = "", origin: str = ""):
        from fastapi.responses import StreamingResponse
        from tongflow import serve_stream_from_spec

        return StreamingResponse(
            serve_stream_from_spec(
                origin, taskId, token, __file__,
                invoke=lambda m, inp: getattr(self, m).local(inp),
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Access-Control-Allow-Origin": "*"},
        )



@deploy
@app.cls(
    scaledown_window=2,
    image=image,
    gpu="L4",
    volumes={"/models": volume},
)
class TranscribeWithTimestamps:
    """ASR + Qwen3-ForcedAligner for per-token timestamps."""

    @modal.enter()
    def load(self):
        self.asr = Qwen3ASRModel.from_pretrained(
            ASR_MODEL_DIR,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            forced_aligner=ALIGNER_MODEL_DIR,
            forced_aligner_kwargs=dict(
                dtype=torch.bfloat16,
                device_map="cuda:0",
                attn_implementation="flash_attention_2",
            ),
            max_inference_batch_size=32,
            max_new_tokens=512,
        )

    def _transcribe_from_bytes(
        self,
        audio_bytes: bytes,
        filename: str = "media.bin",
        context: str = "",
        language: Optional[str] = None,
        max_new_tokens: int = 512,
    ) -> dict:
        self.asr.max_new_tokens = max_new_tokens
        lang = language if (language and str(language).strip()) else None
        with _asr_audio_input_from_bytes(audio_bytes, filename) as asr_src:
            results = self.asr.transcribe(
                audio=asr_src,
                context=context,
                language=lang,
                return_time_stamps=True,
            )
        r = results[0]
        return {
            "language": r.language,
            "text": r.text,
            "time_stamps": _align_result_to_list(r.time_stamps),
        }

    @modal.method()
    def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "media.bin",
        context: str = "",
        language: Optional[str] = None,
        max_new_tokens: int = 512,
    ) -> dict:
        return self._transcribe_from_bytes(
            audio_bytes=audio_bytes,
            filename=filename,
            context=context,
            language=language,
            max_new_tokens=max_new_tokens,
        )

    @modal.method()
    @node_slot(NodeSlots.TRANSCRIBE_TIMESTAMP)
    def transcribe_timestamp_openflow(
        self,
        input: TranscribeTimestampInput,
    ) -> TranscribeTimestampOutput:
        if input.audio is None:
            return TranscribeTimestampOutput(
                success=False, error="Missing `audio` Asset"
            )
        out = self._transcribe_from_bytes(
            audio_bytes=prompt_media_to_bytes(input.audio),
            filename=input.audio.filename or "media.bin",
            context=input.context or "",
            language=input.language or None,
            max_new_tokens=int(input.max_new_tokens)
            if input.max_new_tokens is not None
            else 512,
        )
        time_stamps_raw = out.get("time_stamps") or []
        time_stamps = [
            TranscribeTimestampOutputRootTimeStampsItem(
                start=float(item["start"]),
                end=float(item["end"]),
                text=str(item["text"]),
            )
            for item in time_stamps_raw
        ]
        return TranscribeTimestampOutput(
            success=True,
            text=str(out.get("text") or ""),
            time_stamps=time_stamps,
        )

    @modal.fastapi_endpoint(method="GET", label=f"{Path(__file__).resolve().parent.name}-transcribewithtimestamps-serve")
    def serve(self, taskId: str = "", token: str = "", origin: str = ""):
        from fastapi.responses import StreamingResponse
        from tongflow import serve_stream_from_spec

        return StreamingResponse(
            serve_stream_from_spec(
                origin, taskId, token, __file__,
                invoke=lambda m, inp: getattr(self, m).local(inp),
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Access-Control-Allow-Origin": "*"},
        )

