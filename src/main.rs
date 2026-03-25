use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};
use seisrefine::{
    IngestOptions, InterpMethod, SectionAxis, SeisGeometryOptions, UpscaleOptions, ingest_segy,
    inspect_segy, render_section_csv, run_validation, upscale_store,
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
        #[arg(long)]
        inline_byte: Option<u16>,
        #[arg(long, value_enum, default_value_t = HeaderTypeArg::I32)]
        inline_type: HeaderTypeArg,
        #[arg(long)]
        crossline_byte: Option<u16>,
        #[arg(long, value_enum, default_value_t = HeaderTypeArg::I32)]
        crossline_type: HeaderTypeArg,
        #[arg(long)]
        third_axis_byte: Option<u16>,
        #[arg(long, value_enum, default_value_t = HeaderTypeArg::I32)]
        third_axis_type: HeaderTypeArg,
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

#[derive(Debug, Clone, Copy, ValueEnum)]
enum HeaderTypeArg {
    I16,
    I32,
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
            inline_byte,
            inline_type,
            crossline_byte,
            crossline_type,
            third_axis_byte,
            third_axis_type,
        } => {
            let handle = ingest_segy(
                input,
                output,
                IngestOptions {
                    chunk_shape: parse_chunk_shape(&chunk),
                    geometry: build_ingest_geometry(
                        inline_byte,
                        inline_type,
                        crossline_byte,
                        crossline_type,
                        third_axis_byte,
                        third_axis_type,
                    ),
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

fn build_ingest_geometry(
    inline_byte: Option<u16>,
    inline_type: HeaderTypeArg,
    crossline_byte: Option<u16>,
    crossline_type: HeaderTypeArg,
    third_axis_byte: Option<u16>,
    third_axis_type: HeaderTypeArg,
) -> SeisGeometryOptions {
    let mut geometry = SeisGeometryOptions::default();
    geometry.header_mapping.inline_3d =
        inline_byte.map(|start_byte| header_field("INLINE_3D", start_byte, inline_type));
    geometry.header_mapping.crossline_3d =
        crossline_byte.map(|start_byte| header_field("CROSSLINE_3D", start_byte, crossline_type));
    geometry.third_axis_field =
        third_axis_byte.map(|start_byte| header_field("THIRD_AXIS", start_byte, third_axis_type));
    geometry
}

fn header_field(
    name: &'static str,
    start_byte: u16,
    value_type: HeaderTypeArg,
) -> sgyx::HeaderField {
    match value_type {
        HeaderTypeArg::I16 => sgyx::HeaderField::new_i16(name, start_byte),
        HeaderTypeArg::I32 => sgyx::HeaderField::new_i32(name, start_byte),
    }
}
