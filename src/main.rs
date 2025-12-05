use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, anyhow, bail};
use async_openai::{
    Client,
    config::OpenAIConfig,
    types::audio::{AudioInput, CreateTranscriptionRequestArgs},
};
use clap::{Parser, Subcommand};
use serde::Deserialize;
use tokio::runtime::Runtime;

#[derive(Parser, Debug)]
#[command(author, version, about = "Inspect and chunk audio files with ffmpeg")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Display ffmpeg metadata for an input file
    Inspect {
        /// Audio file to inspect
        input: PathBuf,
    },
    /// Chunk an audio file into ≤ max-size segments using metadata-derived windows
    Chunk {
        /// Audio file to chunk
        input: PathBuf,
        /// Maximum chunk size (in megabytes)
        #[arg(long, default_value_t = 25.0)]
        max_size_mb: f64,
    },
    /// Split an already compliant chunk into N sequential parts
    Split {
        /// Chunk to split further
        input: PathBuf,
        /// Number of parts to create
        #[arg(long)]
        parts: usize,
    },
    /// Transcribe a chunked audio file using OpenAI
    Transcribe {
        /// Audio chunk to transcribe
        input: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Inspect { input } => inspect_audio(&input),
        Commands::Chunk { input, max_size_mb } => chunk_audio(&input, max_size_mb),
        Commands::Split { input, parts } => split_chunk(&input, parts),
        Commands::Transcribe { input } => transcribe_chunk(&input),
    }
}

fn inspect_audio(input: &Path) -> Result<()> {
    ensure_input_exists(input)?;
    let output = Command::new("ffmpeg")
        .arg("-hide_banner")
        .arg("-i")
        .arg(input)
        .arg("-f")
        .arg("null")
        .arg("-")
        .output()
        .with_context(|| "failed to run ffmpeg, is it installed and on PATH?")?;

    if !output.status.success() {
        bail!(
            "ffmpeg returned a non-zero status while inspecting the file:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    println!("{}", String::from_utf8_lossy(&output.stderr));
    Ok(())
}

fn chunk_audio(input: &Path, max_size_mb: f64) -> Result<()> {
    if max_size_mb <= 0.0 {
        bail!("max_size_mb must be greater than zero");
    }

    ensure_input_exists(input)?;
    let metadata = fetch_audio_metadata(input)?;
    let plan = calculate_chunk_plan(
        metadata.duration_seconds,
        metadata.bitrate_kbps,
        max_size_mb,
    )?;

    let parent = input.parent().unwrap_or_else(|| Path::new("."));
    let base_name = input
        .file_stem()
        .map(|stem| stem.to_string_lossy().to_string())
        .unwrap_or_else(|| "chunk".to_string());
    let extension = input
        .extension()
        .map(|ext| format!(".{}", ext.to_string_lossy()))
        .unwrap_or_default();

    for (index, (start, duration)) in plan.iter().enumerate() {
        let output_name = format!("{base_name}_chunk{index:03}{extension}");
        let output_path = parent.join(&output_name);
        let start_arg = format!("{start:.3}");
        let duration_arg = format!("{duration:.3}");

        let status = Command::new("ffmpeg")
            .arg("-hide_banner")
            .arg("-loglevel")
            .arg("error")
            .arg("-y")
            .arg("-i")
            .arg(input)
            .arg("-ss")
            .arg(&start_arg)
            .arg("-t")
            .arg(&duration_arg)
            .arg("-c")
            .arg("copy")
            .arg(&output_path)
            .status()
            .with_context(|| format!("failed to chunk file while creating {output_name}"))?;

        if !status.success() {
            bail!("ffmpeg failed to create {output_name}");
        }

        println!(
            "Created {output_name} (start: {:.3}s, duration: {:.3}s)",
            start, duration
        );
    }

    Ok(())
}

fn split_chunk(input: &Path, parts: usize) -> Result<()> {
    if parts < 2 {
        bail!("parts must be at least 2");
    }

    ensure_input_exists(input)?;
    let metadata = fetch_audio_metadata(input)?;
    ensure_chunk_ready_for_split(input, metadata.duration_seconds)?;

    let plan = calculate_equal_split_plan(metadata.duration_seconds, parts)?;

    let parent = input.parent().unwrap_or_else(|| Path::new("."));
    let base_name = input
        .file_stem()
        .map(|stem| stem.to_string_lossy().to_string())
        .unwrap_or_else(|| "chunk".to_string());
    let extension = input
        .extension()
        .map(|ext| format!(".{}", ext.to_string_lossy()))
        .unwrap_or_default();

    for (index, (start, duration)) in plan.iter().enumerate() {
        let human_index = index + 1;
        let output_name = format!("{base_name}_part{human_index:03}{extension}");
        let output_path = parent.join(&output_name);
        let start_arg = format!("{start:.3}");
        let duration_arg = format!("{duration:.3}");

        let status = Command::new("ffmpeg")
            .arg("-hide_banner")
            .arg("-loglevel")
            .arg("error")
            .arg("-y")
            .arg("-i")
            .arg(input)
            .arg("-ss")
            .arg(&start_arg)
            .arg("-t")
            .arg(&duration_arg)
            .arg("-c")
            .arg("copy")
            .arg(&output_path)
            .status()
            .with_context(|| format!("failed to split file while creating {output_name}"))?;

        if !status.success() {
            bail!("ffmpeg failed to create {output_name}");
        }

        ensure_chunk_within_limit(&output_path)
            .with_context(|| format!("{output_name} exceeded the 25 MB limit"))?;
        ensure_chunk_duration_within_limit(&output_path)
            .with_context(|| format!("{output_name} exceeded the 1400 second limit"))?;

        println!(
            "Created {output_name} (start: {:.3}s, duration: {:.3}s)",
            start, duration
        );
    }

    Ok(())
}

const MAX_CHUNK_BYTES: u64 = 25 * 1024 * 1024;
const OPENAI_MAX_TRANSCRIPTION_DURATION_SECONDS: f64 = 1400.0;
const CHUNK_DURATION_BUFFER_SECONDS: f64 = 100.0;
const PLANNED_MAX_CHUNK_DURATION_SECONDS: f64 =
    OPENAI_MAX_TRANSCRIPTION_DURATION_SECONDS - CHUNK_DURATION_BUFFER_SECONDS;

fn transcribe_chunk(input: &Path) -> Result<()> {
    ensure_input_exists(input)?;
    ensure_chunk_within_limit(input)?;
    ensure_chunk_duration_within_limit(input)?;
    let api_key = load_openai_api_key()?;
    let transcript = transcribe_chunk_with_openai(input, api_key)?;

    let output_path = transcript_output_path(input);
    std::fs::write(&output_path, transcript).with_context(|| {
        format!(
            "failed to write transcript to '{}'",
            output_path.to_string_lossy()
        )
    })?;

    println!("Transcript saved to '{}'", output_path.to_string_lossy());
    Ok(())
}

fn ensure_chunk_within_limit(path: &Path) -> Result<()> {
    let metadata = std::fs::metadata(path)
        .with_context(|| format!("failed to read metadata for '{}'", path.display()))?;
    if metadata.len() > MAX_CHUNK_BYTES {
        bail!(
            "chunk '{}' is larger than the 25 MB limit",
            path.to_string_lossy()
        );
    }

    Ok(())
}

fn ensure_chunk_duration_within_limit(path: &Path) -> Result<()> {
    let metadata = fetch_audio_metadata(path)?;
    if metadata.duration_seconds > OPENAI_MAX_TRANSCRIPTION_DURATION_SECONDS {
        bail!(
            "chunk '{}' is longer than the {} second limit for transcription",
            path.to_string_lossy(),
            OPENAI_MAX_TRANSCRIPTION_DURATION_SECONDS as u32
        );
    }
    Ok(())
}

fn ensure_chunk_ready_for_split(path: &Path, duration_seconds: f64) -> Result<()> {
    let metadata = std::fs::metadata(path)
        .with_context(|| format!("failed to read metadata for '{}'", path.display()))?;

    if metadata.len() > MAX_CHUNK_BYTES
        || duration_seconds > OPENAI_MAX_TRANSCRIPTION_DURATION_SECONDS
    {
        let display = path.to_string_lossy();
        bail!(
            "input '{}' exceeds the chunk limits; run `audio_splitter_cli chunk {}` first",
            display,
            display
        );
    }

    Ok(())
}

fn load_openai_api_key() -> Result<String> {
    let value = std::env::var("OPENAI_API_KEY")
        .with_context(|| "OPENAI_API_KEY environment variable is required for transcription")?;

    if value.trim().is_empty() {
        bail!("OPENAI_API_KEY cannot be empty");
    }

    Ok(value)
}

fn transcribe_chunk_with_openai(chunk_path: &Path, api_key: String) -> Result<String> {
    let file_name = chunk_path
        .file_name()
        .ok_or_else(|| anyhow!("chunk file '{}' has no valid name", chunk_path.display()))?
        .to_string_lossy()
        .to_string();

    let bytes = std::fs::read(chunk_path)
        .with_context(|| format!("failed to read '{}'", chunk_path.display()))?;

    let runtime = Runtime::new().context("failed to start tokio runtime")?;
    runtime
        .block_on(async move {
            let config = OpenAIConfig::new().with_api_key(api_key);
            let client = Client::with_config(config);

            let request = CreateTranscriptionRequestArgs::default()
                .model("gpt-4o-transcribe")
                .file(AudioInput::from_vec_u8(file_name, bytes))
                .build()?;

            let response = client.audio().transcription().create(request).await?;

            Ok::<_, async_openai::error::OpenAIError>(response.text)
        })
        .context("failed to run transcription request on the async runtime")
}

fn transcript_output_path(input: &Path) -> PathBuf {
    let parent = input.parent().unwrap_or_else(|| Path::new("."));
    let file_name = input
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| "transcript".to_string());
    parent.join(format!("{file_name}.txt"))
}

fn ensure_input_exists(path: &Path) -> Result<()> {
    if path.exists() {
        return Ok(());
    }

    bail!("input file '{}' was not found", path.to_string_lossy());
}

fn fetch_audio_metadata(input: &Path) -> Result<AudioMetadata> {
    let output = Command::new("ffprobe")
        .arg("-v")
        .arg("error")
        .arg("-select_streams")
        .arg("a:0")
        .arg("-show_entries")
        .arg("format=duration,bit_rate:stream=bit_rate")
        .arg("-of")
        .arg("json")
        .arg(input)
        .output()
        .with_context(|| "failed to run ffprobe, is it installed and on PATH?")?;

    if !output.status.success() {
        bail!(
            "ffprobe returned a non-zero status:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let parsed: FfprobeOutput = serde_json::from_slice(&output.stdout)
        .with_context(|| "failed to parse ffprobe JSON output")?;

    let duration_seconds = parsed
        .format
        .as_ref()
        .and_then(|f| f.duration.as_deref())
        .ok_or_else(|| anyhow!("duration missing from ffprobe output"))?
        .parse::<f64>()
        .with_context(|| "failed to parse duration from ffprobe output")?;

    let bit_rate_bits = parsed
        .format
        .as_ref()
        .and_then(|f| f.bit_rate.as_deref())
        .or_else(|| {
            parsed
                .streams
                .as_ref()
                .and_then(|streams| streams.first())
                .and_then(|stream| stream.bit_rate.as_deref())
        })
        .ok_or_else(|| anyhow!("bitrate missing from ffprobe output"))?
        .parse::<f64>()
        .with_context(|| "failed to parse bitrate from ffprobe output")?;

    let bitrate_kbps = bit_rate_bits / 1000.0;

    Ok(AudioMetadata {
        duration_seconds,
        bitrate_kbps,
    })
}

#[derive(Debug)]
struct AudioMetadata {
    duration_seconds: f64,
    bitrate_kbps: f64,
}

#[derive(Debug, Deserialize)]
struct FfprobeOutput {
    streams: Option<Vec<FfprobeStream>>,
    format: Option<FfprobeFormat>,
}

#[derive(Debug, Deserialize)]
struct FfprobeStream {
    #[serde(rename = "bit_rate")]
    bit_rate: Option<String>,
}

#[derive(Debug, Deserialize)]
struct FfprobeFormat {
    duration: Option<String>,
    #[serde(rename = "bit_rate")]
    bit_rate: Option<String>,
}

fn calculate_equal_split_plan(duration_seconds: f64, parts: usize) -> Result<Vec<(f64, f64)>> {
    if duration_seconds <= 0.0 {
        bail!("duration_seconds must be greater than zero");
    }
    if parts < 2 {
        bail!("parts must be at least 2");
    }

    let mut plan = Vec::with_capacity(parts);
    let mut start = 0.0;
    for i in 0..parts {
        let remaining = duration_seconds - start;
        let segments_left = parts - i;
        let duration = if segments_left == 1 {
            remaining
        } else {
            remaining / segments_left as f64
        };
        plan.push((start, duration));
        start += duration;
    }

    Ok(plan)
}

/// Calculate start timestamps and durations so each chunk stays within the maximum size (in megabytes).
pub fn calculate_chunk_plan(
    duration_seconds: f64,
    bitrate_kbps: f64,
    max_size_mb: f64,
) -> Result<Vec<(f64, f64)>> {
    if duration_seconds <= 0.0 {
        bail!("duration_seconds must be greater than zero");
    }
    if bitrate_kbps <= 0.0 {
        bail!("bitrate_kbps must be greater than zero");
    }
    if max_size_mb <= 0.0 {
        bail!("max_size_mb must be greater than zero");
    }

    const SAFETY_MARGIN: f64 = 0.94;
    let bits_per_second = bitrate_kbps * 1000.0;
    let max_bits_per_chunk = max_size_mb * 1024.0 * 1024.0 * 8.0;
    // Use a conservative margin so container overhead cannot push a chunk past the limit.
    let safe_bits_per_chunk = max_bits_per_chunk * SAFETY_MARGIN;
    let bits_based_duration = (safe_bits_per_chunk / bits_per_second).floor();
    let chunk_duration = bits_based_duration.min(PLANNED_MAX_CHUNK_DURATION_SECONDS);

    if chunk_duration < 1.0 {
        bail!("calculated chunk duration is less than one second; adjust inputs");
    }

    let mut plan = Vec::new();
    let mut start = 0.0;
    while start < duration_seconds {
        let remaining = duration_seconds - start;
        let duration = chunk_duration.min(remaining);
        plan.push((start, duration));
        start += duration;
    }

    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::{calculate_chunk_plan, calculate_equal_split_plan};

    #[test]
    fn splits_into_expected_chunk_lengths() {
        let plan = calculate_chunk_plan(3600.0, 228.0, 25.0).unwrap();
        assert_eq!(plan.len(), 5);
        assert!((plan[0].0 - 0.0).abs() < 1e-6);
        assert!((plan[0].1 - 864.0).abs() < 1e-6);
        assert!((plan[3].0 - 2592.0).abs() < 1e-6);
        assert!((plan[3].1 - 864.0).abs() < 1e-6);
        assert!((plan[4].0 - 3456.0).abs() < 1e-6);
        assert!((plan[4].1 - 144.0).abs() < 1e-6);
    }

    #[test]
    fn rejects_invalid_inputs() {
        assert!(calculate_chunk_plan(0.0, 228.0, 25.0).is_err());
        assert!(calculate_chunk_plan(10.0, 0.0, 25.0).is_err());
        assert!(calculate_chunk_plan(10.0, 228.0, 0.0).is_err());
    }

    #[test]
    fn caps_chunk_duration_at_transcription_limit() {
        let plan = calculate_chunk_plan(4000.0, 128.0, 25.0).unwrap();
        assert_eq!(plan.len(), 4);
        assert!((plan[0].1 - 1300.0).abs() < 1e-6);
        assert!((plan[1].1 - 1300.0).abs() < 1e-6);
        assert!((plan[2].1 - 1300.0).abs() < 1e-6);
        assert!((plan[3].1 - 100.0).abs() < 1e-6);
    }

    #[test]
    fn split_plan_divides_duration_evenly_with_remainder() {
        let plan = calculate_equal_split_plan(100.0, 3).unwrap();
        assert_eq!(plan.len(), 3);
        assert!((plan[0].0 - 0.0).abs() < 1e-6);
        assert!((plan[0].1 - (100.0 / 3.0)).abs() < 1e-6);
        assert!((plan[1].0 - (100.0 / 3.0)).abs() < 1e-6);
        assert!((plan[1].1 - (100.0 / 3.0)).abs() < 1e-6);
        assert!((plan[2].0 - (200.0 / 3.0)).abs() < 1e-6);
        assert!((plan[2].1 - (100.0 - (200.0 / 3.0))).abs() < 1e-6);
    }

    #[test]
    fn split_plan_rejects_invalid_requests() {
        assert!(calculate_equal_split_plan(0.0, 3).is_err());
        assert!(calculate_equal_split_plan(10.0, 1).is_err());
    }
}
