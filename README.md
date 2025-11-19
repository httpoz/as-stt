# audio_splitter_cli

`audio_splitter_cli` is a Rust-powered helper for prepping long-form recordings for OpenAI's speech-to-text tooling. It wraps `ffmpeg`, `ffprobe`, and the official OpenAI client so you can understand, split, and optionally transcribe audio from one place.

## Capabilities
- Inspect an input file with `audio_splitter_cli inspect <file>` to print the `ffmpeg` metadata (codec, duration, bitrate) before doing any edits.
- Chunk recordings with `audio_splitter_cli chunk <file> --max-size 25` (default 25 MB) so each output stays under the Whisper upload limit.
- Transcribe a chunk via `audio_splitter_cli transcribe <chunk>` once it is under 25 MB and you have the `OPENAI_API_KEY` environment variable configured.

## Requirements
- Rust toolchain (for building/running the CLI with `cargo run …`).
- `ffmpeg` and `ffprobe` accessible on `PATH`.
- `OPENAI_API_KEY` only when using the `transcribe` command.

Expected behaviors, edge cases, and acceptance criteria are documented in the Gherkin specs under `features/`. Update those files first whenever you change how the CLI should work.
