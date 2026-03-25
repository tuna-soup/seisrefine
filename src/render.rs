use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::error::SeisRefineError;
use crate::store::{load_array, open_store};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SectionAxis {
    Inline,
    Xline,
}

pub fn render_section_csv(
    store_root: impl AsRef<Path>,
    axis: SectionAxis,
    index: usize,
    output_path: impl AsRef<Path>,
) -> Result<(), SeisRefineError> {
    let handle = open_store(store_root)?;
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
            if index >= handle.manifest.shape[0] {
                return Err(SeisRefineError::InvalidSectionIndex {
                    index,
                    len: handle.manifest.shape[0],
                });
            }

            for xline in 0..handle.manifest.shape[1] {
                write!(file, "{}", handle.manifest.axes.xlines[xline])?;
                for sample in 0..handle.manifest.shape[2] {
                    write!(file, ",{}", array[[index, xline, sample]])?;
                }
                writeln!(file)?;
            }
        }
        SectionAxis::Xline => {
            if index >= handle.manifest.shape[1] {
                return Err(SeisRefineError::InvalidSectionIndex {
                    index,
                    len: handle.manifest.shape[1],
                });
            }

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
