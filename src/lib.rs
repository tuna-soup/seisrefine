mod error;
mod ingest;
mod metadata;
mod preflight;
mod render;
mod store;
mod upscale;
mod validation;

pub use error::SeisRefineError;
pub use seisdomain_core::{
    DatasetId, InterpretationPoint, ProcessingParameters, SectionAxis, SectionRequest,
    SectionTileRequest, VolumeDescriptor,
};
pub use seisdomain_views::{PreviewView, SectionView};
pub use ingest::{
    IngestOptions, SeisGeometryOptions, SourceVolume, SparseSurveyPolicy, ingest_segy,
    load_source_volume, load_source_volume_with_options,
};
pub use metadata::{
    DatasetKind, DerivedFrom, GeometryProvenance, HeaderFieldSpec, InterpMethod,
    RegularizationProvenance, SourceIdentity, StoreManifest, VolumeAxes,
};
pub use preflight::{
    PreflightAction, PreflightGeometry, SurveyPreflight, preflight_segy,
};
pub use render::{render_section_csv, render_section_csv_for_request};
pub use store::{
    ARRAY_PATH, OCCUPANCY_PATH, StoreHandle, describe_store, load_array, load_occupancy,
    open_store, section_view,
};
pub use upscale::{UpscaleOptions, upscale_2x, upscale_cubic_2x, upscale_linear_2x, upscale_store};
pub use validation::{
    ValidationDatasetReport, ValidationMethodReport, ValidationMetrics, ValidationOptions,
    ValidationSummary, run_validation, validate_dataset,
};

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
    pub warnings: Vec<String>,
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
        warnings: summary
            .warnings
            .iter()
            .map(|warning| format!("{warning:?}"))
            .collect(),
    })
}
