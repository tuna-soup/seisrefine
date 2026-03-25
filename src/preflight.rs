use std::path::Path;

use serde::Serialize;

use crate::error::SeisRefineError;
use crate::ingest::{IngestOptions, geometry_classification_label, header_field_spec};
use crate::{SegyInspection, inspect_segy};

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PreflightAction {
    DirectDenseIngest,
    RegularizeSparseSurvey,
    ReviewGeometryMapping,
    UnsupportedInV1,
}

#[derive(Debug, Clone, Serialize)]
pub struct SurveyPreflight {
    pub inspection: SegyInspection,
    pub geometry: PreflightGeometry,
    pub recommended_action: PreflightAction,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PreflightGeometry {
    pub classification: String,
    pub inline_field: crate::HeaderFieldSpec,
    pub crossline_field: crate::HeaderFieldSpec,
    pub third_axis_field: Option<crate::HeaderFieldSpec>,
    pub inline_count: usize,
    pub crossline_count: usize,
    pub third_axis_count: usize,
    pub observed_trace_count: usize,
    pub unique_coordinate_count: usize,
    pub expected_trace_count: usize,
    pub completeness_ratio: f64,
    pub missing_bin_count: usize,
    pub duplicate_coordinate_count: usize,
}

pub fn preflight_segy(
    path: impl AsRef<Path>,
    options: &IngestOptions,
) -> Result<SurveyPreflight, SeisRefineError> {
    let path = path.as_ref();
    let inspection = inspect_segy(path)?;
    let reader = sgyx::open(
        path,
        sgyx::ReaderOptions {
            validation_mode: options.validation_mode,
            header_mapping: options.geometry.header_mapping.clone(),
            ..sgyx::ReaderOptions::default()
        },
    )?;
    let report = reader.analyze_geometry(sgyx::GeometryOptions {
        third_axis_field: options.geometry.third_axis_field,
        ..sgyx::GeometryOptions::default()
    })?;

    let recommended_action = match report.classification {
        sgyx::GeometryClassification::RegularDense => PreflightAction::DirectDenseIngest,
        sgyx::GeometryClassification::RegularSparse => PreflightAction::RegularizeSparseSurvey,
        sgyx::GeometryClassification::DuplicateCoordinates
        | sgyx::GeometryClassification::AmbiguousMapping => PreflightAction::ReviewGeometryMapping,
        sgyx::GeometryClassification::NonCartesian => PreflightAction::UnsupportedInV1,
    };

    let mut notes = Vec::new();
    match report.classification {
        sgyx::GeometryClassification::RegularDense => {
            notes.push("Survey is already dense under the resolved geometry mapping.".to_string());
        }
        sgyx::GeometryClassification::RegularSparse => {
            notes.push(
                "Survey is sparse on an otherwise regular inline/xline grid and can be regularized explicitly."
                    .to_string(),
            );
            if options.geometry.third_axis_field.is_some() || !report.third_axis_values.is_empty() {
                notes.push(
                    "Current v1 sparse regularization only supports post-stack surveys without an explicit third axis."
                        .to_string(),
                );
            }
        }
        sgyx::GeometryClassification::DuplicateCoordinates => {
            notes.push(
                "Duplicate coordinate tuples were observed; a duplicate-resolution policy is required before ingest."
                    .to_string(),
            );
        }
        sgyx::GeometryClassification::AmbiguousMapping => {
            notes.push(
                "The resolved geometry mapping produces both missing bins and duplicate coordinates. Review header selection before ingest."
                    .to_string(),
            );
        }
        sgyx::GeometryClassification::NonCartesian => {
            notes.push(
                "The dataset does not map cleanly to a Cartesian inline/xline grid under the current mapping."
                    .to_string(),
            );
        }
    }

    Ok(SurveyPreflight {
        inspection,
        geometry: PreflightGeometry {
            classification: geometry_classification_label(report.classification).to_string(),
            inline_field: header_field_spec(report.inline_field),
            crossline_field: header_field_spec(report.crossline_field),
            third_axis_field: report.third_axis_field.map(header_field_spec),
            inline_count: report.inline_values.len(),
            crossline_count: report.crossline_values.len(),
            third_axis_count: report.third_axis_values.len(),
            observed_trace_count: report.observed_trace_count,
            unique_coordinate_count: report.unique_coordinate_count,
            expected_trace_count: report.expected_trace_count,
            completeness_ratio: report.completeness_ratio,
            missing_bin_count: report.missing_bin_count,
            duplicate_coordinate_count: report.duplicate_coordinate_count,
        },
        recommended_action,
        notes,
    })
}
