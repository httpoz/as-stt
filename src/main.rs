use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about = "Inspect audio files with ffmpeg")]
struct Cli {
    /// Audio file to inspect
    input: PathBuf,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    inspect_audio(&cli.input)
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

fn ensure_input_exists(path: &Path) -> Result<()> {
    if path.exists() {
        return Ok(());
    }

    bail!("input file '{}' was not found", path.to_string_lossy());
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

    let bits_per_second = bitrate_kbps * 1000.0;
    let max_bits_per_chunk = max_size_mb * 1024.0 * 1024.0 * 8.0;
    let chunk_duration = (max_bits_per_chunk / bits_per_second).floor();

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
    use super::calculate_chunk_plan;

    #[test]
    fn splits_into_expected_chunk_lengths() {
        let plan = calculate_chunk_plan(3600.0, 228.0, 25.0).unwrap();
        assert_eq!(plan.len(), 4);
        assert!((plan[0].0 - 0.0).abs() < 1e-6);
        assert!((plan[0].1 - 919.0).abs() < 1e-6);
        assert!((plan[3].0 - 2757.0).abs() < 1e-6);
        assert!((plan[3].1 - 843.0).abs() < 1e-6);
    }

    #[test]
    fn rejects_invalid_inputs() {
        assert!(calculate_chunk_plan(0.0, 228.0, 25.0).is_err());
        assert!(calculate_chunk_plan(10.0, 0.0, 25.0).is_err());
        assert!(calculate_chunk_plan(10.0, 228.0, 0.0).is_err());
    }
}
