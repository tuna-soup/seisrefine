use std::path::Path;

use ndarray::Array3;
use sgyx::{ValidationMode, inspect_file, open};

use crate::error::SeisRefineError;
use crate::metadata::{DatasetKind, SourceIdentity, StoreManifest, VolumeAxes};
use crate::store::{StoreHandle, create_store};

#[derive(Debug, Clone)]
pub struct SourceVolume {
    pub source: SourceIdentity,
    pub axes: VolumeAxes,
    pub data: Array3<f32>,
}

#[derive(Debug, Clone, Copy)]
pub struct IngestOptions {
    pub chunk_shape: [usize; 3],
    pub validation_mode: ValidationMode,
}

impl Default for IngestOptions {
    fn default() -> Self {
        Self {
            chunk_shape: [16, 16, 64],
            validation_mode: ValidationMode::Strict,
        }
    }
}

pub fn ingest_segy(
    segy_path: impl AsRef<Path>,
    store_root: impl AsRef<Path>,
    options: IngestOptions,
) -> Result<StoreHandle, SeisRefineError> {
    let volume = load_source_volume(segy_path, options.validation_mode)?;
    let shape = [
        volume.data.shape()[0],
        volume.data.shape()[1],
        volume.data.shape()[2],
    ];
    let manifest = StoreManifest {
        version: 1,
        kind: DatasetKind::Source,
        source: volume.source,
        shape,
        chunk_shape: clamp_chunk_shape(options.chunk_shape, shape),
        axes: volume.axes,
        array_path: String::new(),
        created_by: "seisrefine-0.1.0".to_string(),
        derived_from: None,
    };

    create_store(store_root, manifest, &volume.data)
}

pub fn load_source_volume(
    segy_path: impl AsRef<Path>,
    validation_mode: ValidationMode,
) -> Result<SourceVolume, SeisRefineError> {
    let segy_path = segy_path.as_ref();
    let summary = inspect_file(segy_path)?;
    let reader = open(
        segy_path,
        sgyx::ReaderOptions {
            validation_mode,
            ..sgyx::ReaderOptions::default()
        },
    )?;
    let cube = reader.assemble_cube()?;
    if cube.offsets.len() != 1 {
        return Err(SeisRefineError::UnsupportedOffsetCount {
            offset_count: cube.offsets.len(),
        });
    }

    let shape = [cube.ilines.len(), cube.xlines.len(), cube.samples_per_trace];
    let data = Array3::from_shape_vec(shape, cube.data)?;
    Ok(SourceVolume {
        source: SourceIdentity {
            source_path: segy_path.to_path_buf(),
            file_size: summary.file_size,
            trace_count: summary.trace_count,
            samples_per_trace: cube.samples_per_trace,
            sample_interval_us: cube.sample_interval_us,
            sample_format_code: summary.sample_format_code,
        },
        axes: VolumeAxes {
            ilines: cube.ilines.into_iter().map(|value| value as f64).collect(),
            xlines: cube.xlines.into_iter().map(|value| value as f64).collect(),
            sample_axis_ms: cube.sample_axis_ms,
        },
        data,
    })
}

fn clamp_chunk_shape(chunk_shape: [usize; 3], shape: [usize; 3]) -> [usize; 3] {
    [
        chunk_shape[0].max(1).min(shape[0].max(1)),
        chunk_shape[1].max(1).min(shape[1].max(1)),
        chunk_shape[2].max(1).min(shape[2].max(1)),
    ]
}
