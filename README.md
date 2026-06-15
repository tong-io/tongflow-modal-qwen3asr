# tongflow-modal-qwen3asr

Official TongFlow plugin. Speech recognition with **Qwen3-ASR** (`Qwen/Qwen3-ASR-1.7B`, plus `Qwen/Qwen3-ForcedAligner-0.6B` for word timing), running on a GPU via [Modal](https://modal.com).

## Capabilities

- **Speech recognition** (`transcribe`) — transcribe speech from audio or video.
- **Speech recognition with timestamps** (`transcribe-timestamp`) — transcribe with word/segment timing.

## Credentials

Add in TongFlow **Settings** (gear icon, top-right):

| Key | Required | Notes |
| --- | --- | --- |
| `MODAL_TOKEN_ID` | ✅ | Create at [modal.com/settings/tokens](https://modal.com/settings/tokens). |
| `MODAL_TOKEN_SECRET` | ✅ | Paired with `MODAL_TOKEN_ID`. |

On first use the plugin deploys to your Modal account automatically and caches the build. The Qwen3-ASR weights are public — no Hugging Face token required.
