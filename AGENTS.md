# Agent Guide

## Mission
You are working on **audio_splitter_cli**, a Rust utility that helps prep recordings for OpenAI's speech-to-text workflows. Two outcomes matter:

1. Provide an immediate proof-point by showing ffmpeg metadata for any supplied audio file.
2. (Upcoming) Split long recordings into chunks that are small enough (≤25 MB) for OpenAI Whisper uploads so users can process entire sessions automatically.

This repository treats the Gherkin feature files in `features/` as the source of truth for behavior. They describe both the user stories and the observable acceptance criteria you must satisfy.

## How to Work
- Before touching code, read the relevant `.feature` files to understand the intent, inputs, outputs, and error handling the CLI must support.
- When adding or changing behavior, update or extend the feature files first. New functionality is not "real" until its behavior is captured in Gherkin.
- Implement the CLI so it fulfills the scenarios exactly. Treat the scenarios as high-level tests: every step should map to a concrete CLI behavior (argument parsing, ffmpeg invocation, error message, etc.).
- Keep the metadata command as the current priority. Chunking is aspirational but already scoped in `features/audio_chunking.feature`; use it to drive future stories.
- If there’s ever a disagreement between documentation and feature files, defer to the feature files—they capture the committed requirements.
- Prefer the official OpenAI client libraries (e.g., `async-openai`) when integrating with the API so the CLI stays aligned with platform changes.
