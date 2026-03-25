use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DatasetKind {
    Source,
    Derived,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum InterpMethod {
    Linear,
    Cubic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceIdentity {
    pub source_path: PathBuf,
    pub file_size: u64,
    pub trace_count: u64,
    pub samples_per_trace: usize,
    pub sample_interval_us: u16,
    pub sample_format_code: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeAxes {
    pub ilines: Vec<f64>,
    pub xlines: Vec<f64>,
    pub sample_axis_ms: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivedFrom {
    pub parent_store: PathBuf,
    pub method: InterpMethod,
    pub scale: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreManifest {
    pub version: u32,
    pub kind: DatasetKind,
    pub source: SourceIdentity,
    pub shape: [usize; 3],
    pub chunk_shape: [usize; 3],
    pub axes: VolumeAxes,
    pub array_path: String,
    pub created_by: String,
    pub derived_from: Option<DerivedFrom>,
}

impl StoreManifest {
    pub const FILE_NAME: &'static str = "seisrefine.manifest.json";
}
