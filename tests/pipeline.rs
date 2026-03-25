use std::fs;
use std::path::{Path, PathBuf};

use seisrefine::{
    IngestOptions, SectionAxis, ingest_segy, load_array, open_store, render_section_csv,
    upscale_store,
};
use tempfile::tempdir;

fn fixture_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
        .join("sgyx")
        .join("test-data")
        .join(relative)
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
