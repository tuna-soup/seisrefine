use std::fs::File;
use std::io::Write;
use std::path::Path;

use seisdomain_core::{SectionAxis, SectionRequest};

use crate::error::SeisRefineError;
use crate::store::{load_array, open_store};

pub fn render_section_csv(
    store_root: impl AsRef<Path>,
    axis: SectionAxis,
    index: usize,
    output_path: impl AsRef<Path>,
) -> Result<(), SeisRefineError> {
    let handle = open_store(store_root)?;
    handle.section_view(axis, index)?;
    write_section_csv(&handle, axis, index, output_path)
}

pub fn render_section_csv_for_request(
    store_root: impl AsRef<Path>,
    request: &SectionRequest,
    output_path: impl AsRef<Path>,
) -> Result<(), SeisRefineError> {
    let handle = open_store(store_root)?;
    let expected = handle.dataset_id();
    if expected != request.dataset_id {
        return Err(SeisRefineError::DatasetIdMismatch {
            expected: expected.0,
            found: request.dataset_id.0.clone(),
        });
    }
    handle.section_view(request.axis, request.index)?;
    write_section_csv(&handle, request.axis, request.index, output_path)
}

fn write_section_csv(
    handle: &crate::store::StoreHandle,
    axis: SectionAxis,
    index: usize,
    output_path: impl AsRef<Path>,
) -> Result<(), SeisRefineError> {
    let array = load_array(&handle)?;
    let mut file = File::create(output_path)?;
    let sample_axis = &handle.manifest.axes.sample_axis_ms;

    write!(file, "position")?;
    for value in sample_axis {
        write!(file, ",{value}")?;
    }
    writeln!(file)?;

    match axis {
        SectionAxis::Inline => {
            for xline in 0..handle.manifest.shape[1] {
                write!(file, "{}", handle.manifest.axes.xlines[xline])?;
                for sample in 0..handle.manifest.shape[2] {
                    write!(file, ",{}", array[[index, xline, sample]])?;
                }
                writeln!(file)?;
            }
        }
        SectionAxis::Xline => {
            for iline in 0..handle.manifest.shape[0] {
                write!(file, "{}", handle.manifest.axes.ilines[iline])?;
                for sample in 0..handle.manifest.shape[2] {
                    write!(file, ",{}", array[[iline, index, sample]])?;
                }
                writeln!(file)?;
            }
        }
    }

    Ok(())
}
