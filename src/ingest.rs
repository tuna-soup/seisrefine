use std::path::Path;

use ndarray::Array3;
use sgyx::{
    GeometryClassification, GeometryOptions, HeaderField, HeaderMapping, ValidationMode,
    inspect_file, open,
};

use crate::error::SeisRefineError;
use crate::metadata::{
    DatasetKind, GeometryProvenance, HeaderFieldSpec, SourceIdentity, StoreManifest, VolumeAxes,
};
use crate::store::{StoreHandle, create_store};

#[derive(Debug, Clone)]
pub struct SourceVolume {
    pub source: SourceIdentity,
    pub axes: VolumeAxes,
    pub data: Array3<f32>,
}

#[derive(Debug, Clone)]
pub struct SeisGeometryOptions {
    pub header_mapping: HeaderMapping,
    pub third_axis_field: Option<HeaderField>,
}

impl Default for SeisGeometryOptions {
    fn default() -> Self {
        Self {
            header_mapping: HeaderMapping::default(),
            third_axis_field: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct IngestOptions {
    pub chunk_shape: [usize; 3],
    pub validation_mode: ValidationMode,
    pub geometry: SeisGeometryOptions,
}

impl Default for IngestOptions {
    fn default() -> Self {
        Self {
            chunk_shape: [16, 16, 64],
            validation_mode: ValidationMode::Strict,
            geometry: SeisGeometryOptions::default(),
        }
    }
}

pub fn ingest_segy(
    segy_path: impl AsRef<Path>,
    store_root: impl AsRef<Path>,
    options: IngestOptions,
) -> Result<StoreHandle, SeisRefineError> {
    let volume = load_source_volume_with_options(segy_path, &options)?;
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
    load_source_volume_with_options(
        segy_path,
        &IngestOptions {
            validation_mode,
            ..IngestOptions::default()
        },
    )
}

pub fn load_source_volume_with_options(
    segy_path: impl AsRef<Path>,
    options: &IngestOptions,
) -> Result<SourceVolume, SeisRefineError> {
    let segy_path = segy_path.as_ref();
    let summary = inspect_file(segy_path)?;
    let reader = open(
        segy_path,
        sgyx::ReaderOptions {
            validation_mode: options.validation_mode,
            header_mapping: options.geometry.header_mapping.clone(),
            ..sgyx::ReaderOptions::default()
        },
    )?;
    let geometry_report = reader.analyze_geometry(GeometryOptions {
        third_axis_field: options.geometry.third_axis_field,
        ..GeometryOptions::default()
    })?;
    if !matches!(
        geometry_report.classification,
        GeometryClassification::RegularDense
    ) {
        return Err(SeisRefineError::UnsupportedSurveyGeometry {
            classification: geometry_report.classification,
            observed_trace_count: geometry_report.observed_trace_count,
            expected_trace_count: geometry_report.expected_trace_count,
            missing_bin_count: geometry_report.missing_bin_count,
            duplicate_coordinate_count: geometry_report.duplicate_coordinate_count,
        });
    }
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
            geometry: GeometryProvenance {
                inline_field: header_field_spec(reader.header_mapping().inline_3d()),
                crossline_field: header_field_spec(reader.header_mapping().crossline_3d()),
                third_axis_field: options.geometry.third_axis_field.map(header_field_spec),
            },
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

fn header_field_spec(field: HeaderField) -> HeaderFieldSpec {
    HeaderFieldSpec {
        name: field.name.to_string(),
        start_byte: field.start_byte,
        value_type: format!("{:?}", field.value_type),
    }
}
