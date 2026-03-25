use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};
use seisrefine::{
    IngestOptions, InterpMethod, SectionAxis, UpscaleOptions, ingest_segy, inspect_segy,
    render_section_csv, upscale_store,
};

#[derive(Debug, Parser)]
#[command(name = "seisrefine")]
#[command(about = "SEG-Y ingest and conservative trace-volume refinement")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Inspect {
        input: PathBuf,
    },
    Ingest {
        input: PathBuf,
        output: PathBuf,
        #[arg(long, value_delimiter = ',', default_values_t = [16_usize, 16, 64])]
        chunk: Vec<usize>,
    },
    Upscale {
        input: PathBuf,
        output: PathBuf,
        #[arg(long, default_value_t = 2)]
        scale: u8,
        #[arg(long, value_delimiter = ',', default_values_t = [16_usize, 16, 64])]
        chunk: Vec<usize>,
    },
    Render {
        input: PathBuf,
        output: PathBuf,
        #[arg(long, value_enum)]
        axis: AxisArg,
        #[arg(long)]
        index: usize,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum AxisArg {
    Inline,
    Xline,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Command::Inspect { input } => {
            println!("{}", serde_json::to_string_pretty(&inspect_segy(input)?)?);
        }
        Command::Ingest {
            input,
            output,
            chunk,
        } => {
            let handle = ingest_segy(
                input,
                output,
                IngestOptions {
                    chunk_shape: parse_chunk_shape(&chunk),
                    ..IngestOptions::default()
                },
            )?;
            println!("{}", serde_json::to_string_pretty(&handle.manifest)?);
        }
        Command::Upscale {
            input,
            output,
            scale,
            chunk,
        } => {
            let handle = upscale_store(
                input,
                output,
                UpscaleOptions {
                    scale,
                    method: InterpMethod::Linear,
                    chunk_shape: parse_chunk_shape(&chunk),
                },
            )?;
            println!("{}", serde_json::to_string_pretty(&handle.manifest)?);
        }
        Command::Render {
            input,
            output,
            axis,
            index,
        } => {
            render_section_csv(input, axis.into(), index, output)?;
        }
    }

    Ok(())
}

fn parse_chunk_shape(values: &[usize]) -> [usize; 3] {
    match values {
        [a, b, c] => [*a, *b, *c],
        _ => [16, 16, 64],
    }
}

impl From<AxisArg> for SectionAxis {
    fn from(value: AxisArg) -> Self {
        match value {
            AxisArg::Inline => SectionAxis::Inline,
            AxisArg::Xline => SectionAxis::Xline,
        }
    }
}
