use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum SeisRefineError {
    #[error("{0}")]
    Message(String),
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Json(#[from] serde_json::Error),
    #[error("{0}")]
    SgyxInspect(#[from] sgyx::InspectError),
    #[error("{0}")]
    SgyxRead(#[from] sgyx::ReadError),
    #[error("{0}")]
    Zarr(#[from] zarrs::array::ArrayError),
    #[error("{0}")]
    ZarrStorage(#[from] zarrs::storage::StorageError),
    #[error("{0}")]
    InvalidShape(#[from] ndarray::ShapeError),
    #[error("store root already exists: {0}")]
    StoreAlreadyExists(PathBuf),
    #[error("store manifest missing at {0}")]
    MissingManifest(PathBuf),
    #[error(
        "unsupported survey geometry for dense post-stack ingest: classification={classification:?}, observed={observed_trace_count}, expected={expected_trace_count}, missing_bins={missing_bin_count}, duplicate_coordinates={duplicate_coordinate_count}"
    )]
    UnsupportedSurveyGeometry {
        classification: sgyx::GeometryClassification,
        observed_trace_count: usize,
        expected_trace_count: usize,
        missing_bin_count: usize,
        duplicate_coordinate_count: usize,
    },
    #[error("only post-stack cubes are supported in v1; found {offset_count} offsets")]
    UnsupportedOffsetCount { offset_count: usize },
    #[error("unsupported scale factor {scale}; v1 only supports 2x")]
    UnsupportedScale { scale: u8 },
    #[error("dataset is too small for 2x validation after odd cropping: {shape:?}")]
    DatasetTooSmallForValidation { shape: [usize; 3] },
    #[error("invalid section index {index} for axis length {len}")]
    InvalidSectionIndex { index: usize, len: usize },
}
