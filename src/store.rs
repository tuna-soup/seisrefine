use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ndarray::{Array2, Array3, Ix2, Ix3};
use serde_json::json;
use zarrs::array::{Array, ArrayBuilder, data_type};
use zarrs::filesystem::FilesystemStore;
use zarrs::group::GroupBuilder;
use zarrs::storage::ReadableWritableListableStorage;

use crate::error::SeisRefineError;
use crate::metadata::StoreManifest;

pub const ARRAY_PATH: &str = "/amplitude";
pub const OCCUPANCY_PATH: &str = "/occupancy";

#[derive(Debug, Clone)]
pub struct StoreHandle {
    pub root: PathBuf,
    pub manifest: StoreManifest,
}

impl StoreHandle {
    pub fn manifest_path(&self) -> PathBuf {
        self.root.join(StoreManifest::FILE_NAME)
    }
}

pub fn create_store(
    root: impl AsRef<Path>,
    mut manifest: StoreManifest,
    data: &Array3<f32>,
    occupancy: Option<&Array2<u8>>,
) -> Result<StoreHandle, SeisRefineError> {
    let root = root.as_ref().to_path_buf();
    if root.exists() {
        return Err(SeisRefineError::StoreAlreadyExists(root));
    }

    fs::create_dir_all(&root)?;
    manifest.array_path = ARRAY_PATH.to_string();
    manifest.occupancy_array_path = occupancy.map(|_| OCCUPANCY_PATH.to_string());

    let store: ReadableWritableListableStorage = Arc::new(
        FilesystemStore::new(&root).map_err(|error| SeisRefineError::Message(error.to_string()))?,
    );
    GroupBuilder::new()
        .attributes(
            json!({
                "producer": "seisrefine",
                "manifest": StoreManifest::FILE_NAME,
            })
            .as_object()
            .expect("object literal")
            .clone(),
        )
        .build(store.clone(), "/")
        .map_err(|error| SeisRefineError::Message(error.to_string()))?
        .store_metadata()?;

    let array = ArrayBuilder::new(
        manifest
            .shape
            .iter()
            .map(|value| *value as u64)
            .collect::<Vec<_>>(),
        manifest
            .chunk_shape
            .iter()
            .map(|value| *value as u64)
            .collect::<Vec<_>>(),
        data_type::float32(),
        0.0f32,
    )
    .dimension_names(["iline", "xline", "sample"].into())
    .build(store.clone(), ARRAY_PATH)
    .map_err(|error| SeisRefineError::Message(error.to_string()))?;
    array.store_metadata()?;
    array.store_array_subset(
        &[
            0_u64..manifest.shape[0] as u64,
            0_u64..manifest.shape[1] as u64,
            0_u64..manifest.shape[2] as u64,
        ],
        data.to_owned(),
    )?;

    if let Some(occupancy) = occupancy {
        let occupancy_array = ArrayBuilder::new(
            vec![manifest.shape[0] as u64, manifest.shape[1] as u64],
            vec![
                manifest.chunk_shape[0] as u64,
                manifest.chunk_shape[1] as u64,
            ],
            data_type::uint8(),
            0_u8,
        )
        .dimension_names(["iline", "xline"].into())
        .build(store, OCCUPANCY_PATH)
        .map_err(|error| SeisRefineError::Message(error.to_string()))?;
        occupancy_array.store_metadata()?;
        occupancy_array.store_array_subset(
            &[0_u64..manifest.shape[0] as u64, 0_u64..manifest.shape[1] as u64],
            occupancy.to_owned(),
        )?;
    }

    fs::write(
        root.join(StoreManifest::FILE_NAME),
        serde_json::to_vec_pretty(&manifest)?,
    )?;

    Ok(StoreHandle { root, manifest })
}

pub fn open_store(root: impl AsRef<Path>) -> Result<StoreHandle, SeisRefineError> {
    let root = root.as_ref().to_path_buf();
    let manifest_path = root.join(StoreManifest::FILE_NAME);
    if !manifest_path.exists() {
        return Err(SeisRefineError::MissingManifest(manifest_path));
    }

    let manifest = serde_json::from_slice::<StoreManifest>(&fs::read(&manifest_path)?)?;
    Ok(StoreHandle { root, manifest })
}

pub fn load_array(handle: &StoreHandle) -> Result<Array3<f32>, SeisRefineError> {
    let store: ReadableWritableListableStorage = Arc::new(
        FilesystemStore::new(&handle.root)
            .map_err(|error| SeisRefineError::Message(error.to_string()))?,
    );
    let array = Array::open(store, &handle.manifest.array_path)
        .map_err(|error| SeisRefineError::Message(error.to_string()))?;
    let [ilines, xlines, samples] = handle.manifest.shape;
    let data = array.retrieve_array_subset::<ndarray::ArrayD<f32>>(&[
        0_u64..ilines as u64,
        0_u64..xlines as u64,
        0_u64..samples as u64,
    ])?;
    Ok(data.into_dimensionality::<Ix3>()?)
}

pub fn load_occupancy(handle: &StoreHandle) -> Result<Option<Array2<u8>>, SeisRefineError> {
    let Some(path) = handle.manifest.occupancy_array_path.as_deref() else {
        return Ok(None);
    };

    let store: ReadableWritableListableStorage = Arc::new(
        FilesystemStore::new(&handle.root)
            .map_err(|error| SeisRefineError::Message(error.to_string()))?,
    );
    let array =
        Array::open(store, path).map_err(|error| SeisRefineError::Message(error.to_string()))?;
    let [ilines, xlines, _] = handle.manifest.shape;
    let data = array
        .retrieve_array_subset::<ndarray::ArrayD<u8>>(&[
            0_u64..ilines as u64,
            0_u64..xlines as u64,
        ])?
        .into_dimensionality::<Ix2>()?;
    Ok(Some(data))
}
