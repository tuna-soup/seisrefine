use std::fs;
use std::path::{Path, PathBuf};

use ndarray::Array3;
use seisrefine::{
    DatasetId, IngestOptions, InterpMethod, PreflightAction, SectionAxis, SectionRequest,
    SeisGeometryOptions, SeisRefineError, SparseSurveyPolicy, ValidationOptions, describe_store,
    ingest_segy, load_array, load_occupancy, load_source_volume_with_options, open_store,
    preflight_segy, render_section_csv, render_section_csv_for_request, run_validation,
    section_view, upscale_2x, upscale_store,
};
use tempfile::tempdir;

fn fixture_path(relative: &str) -> PathBuf {
    find_dev_root()
        .join("sgyx")
        .join("test-data")
        .join(relative)
}

fn find_dev_root() -> PathBuf {
    let start = Path::new(env!("CARGO_MANIFEST_DIR"))
        .canonicalize()
        .unwrap();
    for ancestor in start.ancestors() {
        if ancestor.join("sgyx").is_dir() {
            return ancestor.to_path_buf();
        }
    }
    panic!("unable to locate sibling sgyx repository from CARGO_MANIFEST_DIR");
}

fn relocate_small_geometry_headers(path: &Path) {
    let mut bytes = fs::read(path).unwrap();
    let first_trace_offset = 3600usize;
    let trace_size = 240 + (50 * 4);

    for trace_index in 0..25 {
        let trace_offset = first_trace_offset + trace_index * trace_size;
        let inline_src = trace_offset + 188;
        let crossline_src = trace_offset + 192;
        let inline_dst = trace_offset + 16;
        let crossline_dst = trace_offset + 24;

        let inline = bytes[inline_src..inline_src + 4].to_vec();
        let crossline = bytes[crossline_src..crossline_src + 4].to_vec();
        bytes[inline_dst..inline_dst + 4].copy_from_slice(&inline);
        bytes[crossline_dst..crossline_dst + 4].copy_from_slice(&crossline);
        bytes[inline_src..inline_src + 4].fill(0);
        bytes[crossline_src..crossline_src + 4].fill(0);
    }

    fs::write(path, bytes).unwrap();
}

fn bytes_per_sample(sample_format_code: u16) -> usize {
    match sample_format_code {
        1 | 2 | 4 | 5 => 4,
        3 => 2,
        8 => 1,
        code => panic!("unsupported sample format code in test fixture helper: {code}"),
    }
}

fn remove_last_trace(src: &Path, dst: &Path) {
    let summary = sgyx::inspect_file(src).unwrap();
    let mut bytes = fs::read(src).unwrap();
    let trace_size =
        240 + summary.samples_per_trace as usize * bytes_per_sample(summary.sample_format_code);
    bytes.truncate(bytes.len() - trace_size);
    fs::write(dst, bytes).unwrap();
}

#[test]
fn ingest_writes_a_zarr_store_and_manifest() {
    let temp = tempdir().unwrap();
    let store_root = temp.path().join("small.zarr");

    let handle = ingest_segy(
        fixture_path("small.sgy"),
        &store_root,
        IngestOptions {
            chunk_shape: [2, 3, 50],
            ..IngestOptions::default()
        },
    )
    .unwrap();

    assert!(store_root.join("zarr.json").exists() || store_root.join(".zgroup").exists());
    assert!(handle.manifest_path().exists());
    assert_eq!(handle.manifest.shape, [5, 5, 50]);
    assert_eq!(handle.manifest.chunk_shape, [2, 3, 50]);
    assert_eq!(handle.manifest.source.geometry.inline_field.start_byte, 189);
    assert_eq!(
        handle.manifest.source.geometry.crossline_field.start_byte,
        193
    );
    assert!(handle.manifest.source.geometry.third_axis_field.is_none());

    let array = load_array(&handle).unwrap();
    assert_eq!(array.shape(), &[5, 5, 50]);
    assert!((array[[0, 0, 0]] - 1.199_999_8).abs() < 1.0e-5);
}

#[test]
fn ingest_rejects_irregular_geometry_with_structured_error() {
    let temp = tempdir().unwrap();
    let store_root = temp.path().join("small-ps.zarr");
    let error = ingest_segy(
        fixture_path("small-ps.sgy"),
        &store_root,
        IngestOptions::default(),
    )
    .unwrap_err();

    assert!(matches!(
        error,
        SeisRefineError::UnsupportedSurveyGeometry { .. }
    ));
}

#[test]
fn preflight_reports_sparse_regular_recommendation() {
    let temp = tempdir().unwrap();
    let sparse_path = temp.path().join("small-sparse.sgy");
    remove_last_trace(&fixture_path("small.sgy"), &sparse_path);

    let preflight = preflight_segy(&sparse_path, &IngestOptions::default()).unwrap();

    assert_eq!(
        preflight.geometry.classification,
        "regular_sparse".to_string()
    );
    assert_eq!(
        preflight.recommended_action,
        PreflightAction::RegularizeSparseSurvey
    );
    assert_eq!(preflight.geometry.missing_bin_count, 1);
}

#[test]
fn ingest_accepts_explicit_header_mapping_for_nonstandard_dense_file() {
    let temp = tempdir().unwrap();
    let segy_path = temp.path().join("small-alt.sgy");
    fs::copy(fixture_path("small.sgy"), &segy_path).unwrap();
    relocate_small_geometry_headers(&segy_path);

    let volume = load_source_volume_with_options(
        &segy_path,
        &IngestOptions {
            geometry: SeisGeometryOptions {
                header_mapping: sgyx::HeaderMapping {
                    inline_3d: Some(sgyx::HeaderField::new_i32("INLINE_3D_ALT", 17)),
                    crossline_3d: Some(sgyx::HeaderField::new_i32("CROSSLINE_3D_ALT", 25)),
                    ..sgyx::HeaderMapping::default()
                },
                third_axis_field: None,
            },
            ..IngestOptions::default()
        },
    )
    .unwrap();

    assert_eq!(volume.data.shape(), &[5, 5, 50]);
    assert_eq!(volume.source.geometry.inline_field.start_byte, 17);
    assert_eq!(volume.source.geometry.crossline_field.start_byte, 25);
}

#[test]
fn ingest_can_regularize_sparse_poststack_with_occupancy_mask() {
    let temp = tempdir().unwrap();
    let sparse_path = temp.path().join("small-sparse.sgy");
    let store_root = temp.path().join("small-sparse.zarr");
    remove_last_trace(&fixture_path("small.sgy"), &sparse_path);

    let handle = ingest_segy(
        &sparse_path,
        &store_root,
        IngestOptions {
            sparse_survey_policy: SparseSurveyPolicy::RegularizeToDense { fill_value: -999.25 },
            ..IngestOptions::default()
        },
    )
    .unwrap();

    let array = load_array(&handle).unwrap();
    let occupancy = load_occupancy(&handle).unwrap().unwrap();
    assert_eq!(array.shape(), &[5, 5, 50]);
    assert_eq!(occupancy.shape(), &[5, 5]);
    assert_eq!(occupancy[[4, 4]], 0);
    assert!((array[[4, 4, 0]] + 999.25).abs() < 1.0e-6);
    assert_eq!(
        handle
            .manifest
            .source
            .regularization
            .as_ref()
            .unwrap()
            .source_classification,
        "regular_sparse"
    );
    assert_eq!(
        handle.manifest.occupancy_array_path.as_deref(),
        Some("/occupancy")
    );
}

#[test]
fn upscale_generates_dense_midpoints_and_preserves_samples() {
    let temp = tempdir().unwrap();
    let source_root = temp.path().join("source.zarr");
    let derived_root = temp.path().join("derived.zarr");

    ingest_segy(
        fixture_path("small.sgy"),
        &source_root,
        IngestOptions::default(),
    )
    .unwrap();
    let derived = upscale_store(&source_root, &derived_root, Default::default()).unwrap();
    let array = load_array(&derived).unwrap();

    assert_eq!(derived.manifest.shape, [9, 9, 50]);
    let source = load_array(&open_store(&source_root).unwrap()).unwrap();
    let midpoint = array[[0, 1, 0]];
    let expected = (source[[0, 0, 0]] + source[[0, 1, 0]]) * 0.5;
    assert!((midpoint - expected).abs() < 1.0e-5);
    assert_eq!(derived.manifest.axes.sample_axis_ms.len(), 50);
}

#[test]
fn render_exports_inline_section_to_csv() {
    let temp = tempdir().unwrap();
    let source_root = temp.path().join("source.zarr");
    let csv_path = temp.path().join("inline.csv");

    ingest_segy(
        fixture_path("small.sgy"),
        &source_root,
        IngestOptions::default(),
    )
    .unwrap();
    render_section_csv(&source_root, SectionAxis::Inline, 0, &csv_path).unwrap();

    let csv = fs::read_to_string(csv_path).unwrap();
    let mut lines = csv.lines();
    let header = lines.next().unwrap();
    assert!(header.starts_with("position,"));
    let first_row = lines.next().unwrap();
    assert!(first_row.starts_with("20,"));
}

#[test]
fn describe_store_returns_shared_volume_descriptor() {
    let temp = tempdir().unwrap();
    let source_root = temp.path().join("source.zarr");

    ingest_segy(
        fixture_path("small.sgy"),
        &source_root,
        IngestOptions::default(),
    )
    .unwrap();

    let descriptor = describe_store(&source_root).unwrap();
    assert_eq!(descriptor.id.0, "source.zarr");
    assert_eq!(descriptor.label, "source");
    assert_eq!(descriptor.shape, [5, 5, 50]);
    assert_eq!(descriptor.chunk_shape, [5, 5, 50]);
}

#[test]
fn section_view_returns_shared_section_view() {
    let temp = tempdir().unwrap();
    let source_root = temp.path().join("source.zarr");

    ingest_segy(
        fixture_path("small.sgy"),
        &source_root,
        IngestOptions::default(),
    )
    .unwrap();

    let view = section_view(&source_root, SectionAxis::Inline, 0).unwrap();
    assert_eq!(view.dataset_id.0, "source.zarr");
    assert_eq!(view.axis, SectionAxis::Inline);
    assert_eq!(view.traces, 5);
    assert_eq!(view.samples, 50);
}

#[test]
fn request_driven_render_rejects_dataset_mismatch() {
    let temp = tempdir().unwrap();
    let source_root = temp.path().join("source.zarr");
    let csv_path = temp.path().join("inline.csv");

    ingest_segy(
        fixture_path("small.sgy"),
        &source_root,
        IngestOptions::default(),
    )
    .unwrap();

    let error = render_section_csv_for_request(
        &source_root,
        &SectionRequest {
            dataset_id: DatasetId("other.zarr".to_string()),
            axis: SectionAxis::Inline,
            index: 0,
        },
        &csv_path,
    )
    .unwrap_err();

    assert!(matches!(error, SeisRefineError::DatasetIdMismatch { .. }));
}

#[test]
fn cubic_matches_linear_on_linear_ramp_midpoints() {
    let input = Array3::from_shape_vec(
        (3, 3, 1),
        vec![
            0.0, 1.0, 2.0, //
            10.0, 11.0, 12.0, //
            20.0, 21.0, 22.0,
        ],
    )
    .unwrap();

    let linear = upscale_2x(&input, InterpMethod::Linear);
    let cubic = upscale_2x(&input, InterpMethod::Cubic);

    assert_eq!(linear.shape(), &[5, 5, 1]);
    assert_eq!(cubic.shape(), &[5, 5, 1]);
    assert!((linear[[0, 1, 0]] - 0.5).abs() < 1.0e-6);
    assert!((cubic[[0, 1, 0]] - 0.5).abs() < 1.0e-6);
    assert!((cubic[[1, 1, 0]] - 5.5).abs() < 1.0e-6);
}

#[test]
fn validation_writes_dataset_and_summary_reports() {
    let temp = tempdir().unwrap();
    let output_dir = temp.path().join("validation");
    let summary = run_validation(ValidationOptions {
        output_dir: output_dir.clone(),
        dataset_paths: vec![fixture_path("small.sgy")],
        validation_mode: sgyx::ValidationMode::Strict,
    })
    .unwrap();

    assert_eq!(summary.dataset_count, 1);
    assert!(output_dir.join("summary.json").exists());
    assert!(output_dir.join("small.json").exists());
}
