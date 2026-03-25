use std::fs;
use std::path::{Path, PathBuf};

use ndarray::Array3;
use seisrefine::{
    IngestOptions, InterpMethod, SectionAxis, ValidationOptions, ingest_segy, load_array,
    open_store, render_section_csv, run_validation, upscale_2x, upscale_store,
};
use tempfile::tempdir;

fn fixture_path(relative: &str) -> PathBuf {
    find_dev_root()
        .join("sgyx")
        .join("test-data")
        .join(relative)
}

fn find_dev_root() -> PathBuf {
    let start = Path::new(env!("CARGO_MANIFEST_DIR")).canonicalize().unwrap();
    for ancestor in start.ancestors() {
        if ancestor.join("sgyx").is_dir() {
            return ancestor.to_path_buf();
        }
    }
    panic!("unable to locate sibling sgyx repository from CARGO_MANIFEST_DIR");
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

    let array = load_array(&handle).unwrap();
    assert_eq!(array.shape(), &[5, 5, 50]);
    assert!((array[[0, 0, 0]] - 1.199_999_8).abs() < 1.0e-5);
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
