mod error;
mod ingest;
mod metadata;
mod render;
mod store;
mod upscale;

pub use error::SeisRefineError;
pub use ingest::{IngestOptions, ingest_segy};
pub use metadata::{
    DatasetKind, DerivedFrom, InterpMethod, SourceIdentity, StoreManifest, VolumeAxes,
};
pub use render::{SectionAxis, render_section_csv};
pub use store::{ARRAY_PATH, StoreHandle, load_array, open_store};
pub use upscale::{UpscaleOptions, upscale_linear_2x, upscale_store};

use std::path::Path;

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct SegyInspection {
    pub file_size: u64,
    pub trace_count: u64,
    pub samples_per_trace: u16,
    pub sample_interval_us: u16,
    pub sample_format_code: u16,
    pub fixed_length_trace: Option<bool>,
    pub endianness: String,
}

pub fn inspect_segy(path: impl AsRef<Path>) -> Result<SegyInspection, SeisRefineError> {
    let summary = sgyx::inspect_file(path)?;
    Ok(SegyInspection {
        file_size: summary.file_size,
        trace_count: summary.trace_count,
        samples_per_trace: summary.samples_per_trace,
        sample_interval_us: summary.sample_interval_us,
        sample_format_code: summary.sample_format_code,
        fixed_length_trace: summary.fixed_length_trace,
        endianness: format!("{:?}", summary.endianness),
    })
}
