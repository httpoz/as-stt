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
