use std::fs;
use std::path::{Path, PathBuf};

use ndarray::{Array3, s};
use serde::Serialize;
use sgyx::ValidationMode;

use crate::SeisRefineError;
use crate::ingest::load_source_volume;
use crate::metadata::InterpMethod;
use crate::upscale::upscale_2x;

#[derive(Debug, Clone)]
pub struct ValidationOptions {
    pub output_dir: PathBuf,
    pub dataset_paths: Vec<PathBuf>,
    pub validation_mode: ValidationMode,
}

#[derive(Debug, Clone, Serialize)]
pub struct ValidationMetrics {
    pub rmse: f64,
    pub mae: f64,
    pub max_abs: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ValidationMethodReport {
    pub method: InterpMethod,
    pub metrics: ValidationMetrics,
}

#[derive(Debug, Clone, Serialize)]
pub struct ValidationDatasetReport {
    pub dataset: String,
    pub source_path: PathBuf,
    pub original_shape: [usize; 3],
    pub evaluated_shape: [usize; 3],
    pub low_res_shape: [usize; 3],
    pub linear: ValidationMethodReport,
    pub cubic: ValidationMethodReport,
    pub cubic_beats_linear_on_rmse: bool,
    pub cubic_beats_linear_on_all_metrics: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ValidationSummary {
    pub dataset_count: usize,
    pub datasets: Vec<ValidationDatasetReport>,
    pub aggregate_linear: ValidationMetrics,
    pub aggregate_cubic: ValidationMetrics,
    pub cubic_beats_linear_on_every_dataset_rmse: bool,
    pub cubic_beats_linear_on_every_dataset_all_metrics: bool,
    pub recommended_default_method: InterpMethod,
}

pub fn run_validation(options: ValidationOptions) -> Result<ValidationSummary, SeisRefineError> {
    fs::create_dir_all(&options.output_dir)?;
    let datasets = if options.dataset_paths.is_empty() {
        discover_default_dataset_paths(options.validation_mode)?
    } else {
        options.dataset_paths
    };

    let mut reports = Vec::with_capacity(datasets.len());
    for path in datasets {
        let report = validate_dataset(&path, options.validation_mode)?;
        let file_name = format!("{}.json", sanitize_name(&report.dataset));
        fs::write(
            options.output_dir.join(file_name),
            serde_json::to_vec_pretty(&report)?,
        )?;
        reports.push(report);
    }

    let summary = build_summary(reports);
    fs::write(
        options.output_dir.join("summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    Ok(summary)
}

pub fn validate_dataset(
    segy_path: impl AsRef<Path>,
    validation_mode: ValidationMode,
) -> Result<ValidationDatasetReport, SeisRefineError> {
    let segy_path = segy_path.as_ref();
    let volume = load_source_volume(segy_path, validation_mode)?;
    let original_shape = shape_of(&volume.data);
    let truth = crop_to_odd_dims(&volume.data);
    let evaluated_shape = shape_of(&truth);
    if evaluated_shape[0] < 3 || evaluated_shape[1] < 3 {
        return Err(SeisRefineError::DatasetTooSmallForValidation {
            shape: evaluated_shape,
        });
    }

    let degraded = thin_spatial_2x(&truth);
    let low_res_shape = shape_of(&degraded);
    let linear_output = upscale_2x(&degraded, InterpMethod::Linear);
    let cubic_output = upscale_2x(&degraded, InterpMethod::Cubic);
    let linear_metrics = compute_metrics(&truth, &linear_output);
    let cubic_metrics = compute_metrics(&truth, &cubic_output);

    Ok(ValidationDatasetReport {
        dataset: dataset_name(segy_path),
        source_path: volume.source.source_path,
        original_shape,
        evaluated_shape,
        low_res_shape,
        linear: ValidationMethodReport {
            method: InterpMethod::Linear,
            metrics: linear_metrics.clone(),
        },
        cubic: ValidationMethodReport {
            method: InterpMethod::Cubic,
            metrics: cubic_metrics.clone(),
        },
        cubic_beats_linear_on_rmse: cubic_metrics.rmse < linear_metrics.rmse,
        cubic_beats_linear_on_all_metrics: cubic_metrics.rmse < linear_metrics.rmse
            && cubic_metrics.mae < linear_metrics.mae
            && cubic_metrics.max_abs < linear_metrics.max_abs,
    })
}

fn discover_default_dataset_paths(
    validation_mode: ValidationMode,
) -> Result<Vec<PathBuf>, SeisRefineError> {
    let root = find_dev_root()?;
    let candidates = vec![
        root.join("sgyx").join("test-data").join("small.sgy"),
        root.join("sgyx").join("test-data").join("f3.sgy"),
        root.join("sgyx").join("test-data").join("long.sgy"),
        root.join("segyio")
            .join("python")
            .join("tutorials")
            .join("viking_small.segy"),
        root.join("sgyx").join("test-data").join("small-lsb.sgy"),
    ];

    let datasets = candidates
        .into_iter()
        .filter(|path| load_source_volume(path, validation_mode).is_ok())
        .collect::<Vec<_>>();

    Ok(datasets)
}

fn find_dev_root() -> Result<PathBuf, SeisRefineError> {
    let start = Path::new(env!("CARGO_MANIFEST_DIR")).canonicalize()?;
    for ancestor in start.ancestors() {
        if ancestor.join("sgyx").is_dir() {
            return Ok(ancestor.to_path_buf());
        }
    }

    Err(SeisRefineError::Message(
        "unable to locate sibling sgyx repository from CARGO_MANIFEST_DIR".to_string(),
    ))
}

fn crop_to_odd_dims(input: &Array3<f32>) -> Array3<f32> {
    let ilines = if input.shape()[0] % 2 == 0 {
        input.shape()[0] - 1
    } else {
        input.shape()[0]
    };
    let xlines = if input.shape()[1] % 2 == 0 {
        input.shape()[1] - 1
    } else {
        input.shape()[1]
    };
    input.slice(s![0..ilines, 0..xlines, ..]).to_owned()
}

fn thin_spatial_2x(input: &Array3<f32>) -> Array3<f32> {
    input.slice(s![..;2, ..;2, ..]).to_owned()
}

fn compute_metrics(truth: &Array3<f32>, estimate: &Array3<f32>) -> ValidationMetrics {
    let mut squared_sum = 0.0_f64;
    let mut abs_sum = 0.0_f64;
    let mut max_abs = 0.0_f64;
    let mut count = 0usize;

    for (truth, estimate) in truth.iter().zip(estimate.iter()) {
        let diff = f64::from(*estimate) - f64::from(*truth);
        let abs = diff.abs();
        squared_sum += diff * diff;
        abs_sum += abs;
        max_abs = max_abs.max(abs);
        count += 1;
    }

    ValidationMetrics {
        rmse: (squared_sum / count as f64).sqrt(),
        mae: abs_sum / count as f64,
        max_abs,
    }
}

fn build_summary(datasets: Vec<ValidationDatasetReport>) -> ValidationSummary {
    let aggregate_linear =
        aggregate_metrics(datasets.iter().map(|dataset| &dataset.linear.metrics));
    let aggregate_cubic = aggregate_metrics(datasets.iter().map(|dataset| &dataset.cubic.metrics));
    let cubic_beats_linear_on_every_dataset_rmse = datasets
        .iter()
        .all(|dataset| dataset.cubic_beats_linear_on_rmse);
    let cubic_beats_linear_on_every_dataset_all_metrics = datasets
        .iter()
        .all(|dataset| dataset.cubic_beats_linear_on_all_metrics);
    let recommended_default_method = if cubic_beats_linear_on_every_dataset_rmse {
        InterpMethod::Cubic
    } else {
        InterpMethod::Linear
    };

    ValidationSummary {
        dataset_count: datasets.len(),
        datasets,
        aggregate_linear,
        aggregate_cubic,
        cubic_beats_linear_on_every_dataset_rmse,
        cubic_beats_linear_on_every_dataset_all_metrics,
        recommended_default_method,
    }
}

fn aggregate_metrics<'a>(
    metrics: impl Iterator<Item = &'a ValidationMetrics>,
) -> ValidationMetrics {
    let mut rmse = 0.0;
    let mut mae = 0.0;
    let mut max_abs = 0.0_f64;
    let mut count = 0usize;
    for metric in metrics {
        rmse += metric.rmse;
        mae += metric.mae;
        max_abs = max_abs.max(metric.max_abs);
        count += 1;
    }

    ValidationMetrics {
        rmse: rmse / count as f64,
        mae: mae / count as f64,
        max_abs,
    }
}

fn shape_of(array: &Array3<f32>) -> [usize; 3] {
    [array.shape()[0], array.shape()[1], array.shape()[2]]
}

fn dataset_name(path: &Path) -> String {
    path.file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("dataset")
        .to_string()
}

fn sanitize_name(name: &str) -> String {
    name.chars()
        .map(|char| {
            if char.is_ascii_alphanumeric() || char == '-' || char == '_' {
                char
            } else {
                '_'
            }
        })
        .collect()
}
