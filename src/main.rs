use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};
use seisrefine::{
    IngestOptions, InterpMethod, SectionAxis, UpscaleOptions, ingest_segy, inspect_segy,
    render_section_csv, run_validation, upscale_store,
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
        #[arg(long, value_enum, default_value_t = MethodArg::Linear)]
        method: MethodArg,
        #[arg(long, value_delimiter = ',', default_values_t = [16_usize, 16, 64])]
        chunk: Vec<usize>,
    },
    Validate {
        output: PathBuf,
        #[arg(long = "input")]
        inputs: Vec<PathBuf>,
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

#[derive(Debug, Clone, Copy, ValueEnum)]
enum MethodArg {
    Linear,
    Cubic,
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
            method,
            chunk,
        } => {
            let handle = upscale_store(
                input,
                output,
                UpscaleOptions {
                    scale,
                    method: method.into(),
                    chunk_shape: parse_chunk_shape(&chunk),
                },
            )?;
            println!("{}", serde_json::to_string_pretty(&handle.manifest)?);
        }
        Command::Validate { output, inputs } => {
            let summary = run_validation(seisrefine::ValidationOptions {
                output_dir: output,
                dataset_paths: inputs,
                validation_mode: sgyx::ValidationMode::Strict,
            })?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
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

impl From<MethodArg> for InterpMethod {
    fn from(value: MethodArg) -> Self {
        match value {
            MethodArg::Linear => InterpMethod::Linear,
            MethodArg::Cubic => InterpMethod::Cubic,
        }
    }
}
