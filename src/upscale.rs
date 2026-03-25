use std::path::Path;

use ndarray::Array3;
use rayon::prelude::*;

use crate::error::SeisRefineError;
use crate::metadata::{DatasetKind, DerivedFrom, InterpMethod, StoreManifest, VolumeAxes};
use crate::store::{StoreHandle, create_store, load_array, open_store};

#[derive(Debug, Clone, Copy)]
pub struct UpscaleOptions {
    pub scale: u8,
    pub method: InterpMethod,
    pub chunk_shape: [usize; 3],
}

impl Default for UpscaleOptions {
    fn default() -> Self {
        Self {
            scale: 2,
            method: InterpMethod::Linear,
            chunk_shape: [16, 16, 64],
        }
    }
}

pub fn upscale_store(
    input_store_root: impl AsRef<Path>,
    output_store_root: impl AsRef<Path>,
    options: UpscaleOptions,
) -> Result<StoreHandle, SeisRefineError> {
    if options.scale != 2 {
        return Err(SeisRefineError::UnsupportedScale {
            scale: options.scale,
        });
    }

    let input = open_store(input_store_root)?;
    let input_array = load_array(&input)?;
    let output_array = upscale_2x(&input_array, options.method);
    let shape = [
        output_array.shape()[0],
        output_array.shape()[1],
        output_array.shape()[2],
    ];
    let manifest = StoreManifest {
        version: 1,
        kind: DatasetKind::Derived,
        source: input.manifest.source.clone(),
        shape,
        chunk_shape: clamp_chunk_shape(options.chunk_shape, shape),
        axes: VolumeAxes {
            ilines: densify_coords(&input.manifest.axes.ilines),
            xlines: densify_coords(&input.manifest.axes.xlines),
            sample_axis_ms: input.manifest.axes.sample_axis_ms.clone(),
        },
        array_path: String::new(),
        created_by: "seisrefine-0.1.0".to_string(),
        derived_from: Some(DerivedFrom {
            parent_store: input.root.clone(),
            method: options.method,
            scale: options.scale,
        }),
    };

    create_store(output_store_root, manifest, &output_array)
}

pub fn upscale_2x(input: &Array3<f32>, method: InterpMethod) -> Array3<f32> {
    upscale_with_method(input, method)
}

pub fn upscale_linear_2x(input: &Array3<f32>) -> Array3<f32> {
    upscale_with_method(input, InterpMethod::Linear)
}

pub fn upscale_cubic_2x(input: &Array3<f32>) -> Array3<f32> {
    upscale_with_method(input, InterpMethod::Cubic)
}

fn upscale_with_method(input: &Array3<f32>, method: InterpMethod) -> Array3<f32> {
    let [ilines, xlines, samples]: [usize; 3] = input.shape().try_into().expect("3D array");
    let out_ilines = if ilines <= 1 { ilines } else { ilines * 2 - 1 };
    let out_xlines = if xlines <= 1 { xlines } else { xlines * 2 - 1 };
    let mut output = vec![0.0_f32; out_ilines * out_xlines * samples];
    let src = input
        .as_slice_memory_order()
        .expect("ingested arrays should be contiguous");

    output
        .par_chunks_mut(out_xlines * samples)
        .enumerate()
        .for_each(|(out_i, row)| {
            for out_x in 0..out_xlines {
                let row_offset = out_x * samples;
                let dst = &mut row[row_offset..row_offset + samples];
                if out_i % 2 == 0 && out_x % 2 == 0 {
                    dst.copy_from_slice(trace_slice(src, xlines, samples, out_i / 2, out_x / 2));
                    continue;
                }

                for sample in 0..samples {
                    dst[sample] =
                        interpolate_sample(src, xlines, samples, out_i, out_x, sample, method);
                }
            }
        });

    Array3::from_shape_vec((out_ilines, out_xlines, samples), output)
        .expect("output shape should be valid")
}

fn trace_slice<'a>(
    src: &'a [f32],
    xlines: usize,
    samples: usize,
    iline: usize,
    xline: usize,
) -> &'a [f32] {
    let index = (iline * xlines + xline) * samples;
    &src[index..index + samples]
}

fn interpolate_sample(
    src: &[f32],
    xlines: usize,
    samples: usize,
    out_i: usize,
    out_x: usize,
    sample: usize,
    method: InterpMethod,
) -> f32 {
    match (out_i % 2, out_x % 2, method) {
        (0, 1, InterpMethod::Linear) => {
            let i = out_i / 2;
            let x = out_x / 2;
            let a = source_value(src, xlines, samples, i as isize, x as isize, sample);
            let b = source_value(src, xlines, samples, i as isize, x as isize + 1, sample);
            (a + b) * 0.5
        }
        (1, 0, InterpMethod::Linear) => {
            let i = out_i / 2;
            let x = out_x / 2;
            let a = source_value(src, xlines, samples, i as isize, x as isize, sample);
            let b = source_value(src, xlines, samples, i as isize + 1, x as isize, sample);
            (a + b) * 0.5
        }
        (1, 1, InterpMethod::Linear) => {
            let i = out_i / 2;
            let x = out_x / 2;
            let a = source_value(src, xlines, samples, i as isize, x as isize, sample);
            let b = source_value(src, xlines, samples, i as isize, x as isize + 1, sample);
            let c = source_value(src, xlines, samples, i as isize + 1, x as isize, sample);
            let d = source_value(src, xlines, samples, i as isize + 1, x as isize + 1, sample);
            (a + b + c + d) * 0.25
        }
        (_, _, InterpMethod::Cubic) => {
            interpolate_cubic_half(src, xlines, samples, out_i, out_x, sample)
        }
        _ => unreachable!(),
    }
}

fn interpolate_cubic_half(
    src: &[f32],
    xlines: usize,
    samples: usize,
    out_i: usize,
    out_x: usize,
    sample: usize,
) -> f32 {
    let ilines = src.len() / (xlines * samples);
    let base_i = (out_i / 2) as isize;
    let base_x = (out_x / 2) as isize;
    match (out_i % 2, out_x % 2) {
        (0, 1) if base_x == 0 || base_x as usize + 1 >= xlines - 1 => {
            let a = source_value(src, xlines, samples, base_i, base_x, sample);
            let b = source_value(src, xlines, samples, base_i, base_x + 1, sample);
            (a + b) * 0.5
        }
        (0, 1) => cubic_half(
            source_value(src, xlines, samples, base_i, base_x - 1, sample),
            source_value(src, xlines, samples, base_i, base_x, sample),
            source_value(src, xlines, samples, base_i, base_x + 1, sample),
            source_value(src, xlines, samples, base_i, base_x + 2, sample),
        ),
        (1, 0) if base_i == 0 || base_i as usize + 1 >= ilines - 1 => {
            let a = source_value(src, xlines, samples, base_i, base_x, sample);
            let b = source_value(src, xlines, samples, base_i + 1, base_x, sample);
            (a + b) * 0.5
        }
        (1, 0) => cubic_half(
            source_value(src, xlines, samples, base_i - 1, base_x, sample),
            source_value(src, xlines, samples, base_i, base_x, sample),
            source_value(src, xlines, samples, base_i + 1, base_x, sample),
            source_value(src, xlines, samples, base_i + 2, base_x, sample),
        ),
        (1, 1)
            if base_i == 0
                || base_x == 0
                || base_i as usize + 1 >= ilines - 1
                || base_x as usize + 1 >= xlines - 1 =>
        {
            let a = source_value(src, xlines, samples, base_i, base_x, sample);
            let b = source_value(src, xlines, samples, base_i, base_x + 1, sample);
            let c = source_value(src, xlines, samples, base_i + 1, base_x, sample);
            let d = source_value(src, xlines, samples, base_i + 1, base_x + 1, sample);
            (a + b + c + d) * 0.25
        }
        (1, 1) => {
            let mut row_values = [0.0_f32; 4];
            for (row_offset, row_value) in row_values.iter_mut().enumerate() {
                let y = base_i + row_offset as isize - 1;
                *row_value = cubic_half(
                    source_value(src, xlines, samples, y, base_x - 1, sample),
                    source_value(src, xlines, samples, y, base_x, sample),
                    source_value(src, xlines, samples, y, base_x + 1, sample),
                    source_value(src, xlines, samples, y, base_x + 2, sample),
                );
            }
            cubic_half(row_values[0], row_values[1], row_values[2], row_values[3])
        }
        _ => unreachable!(),
    }
}

fn source_value(
    src: &[f32],
    xlines: usize,
    samples: usize,
    iline: isize,
    xline: isize,
    sample: usize,
) -> f32 {
    let iline = iline.max(0) as usize;
    let xline = xline.max(0) as usize;
    let max_iline = (src.len() / (xlines * samples)).saturating_sub(1);
    let iline = iline.min(max_iline);
    let xline = xline.min(xlines.saturating_sub(1));
    let index = (iline * xlines + xline) * samples + sample;
    src[index]
}

fn cubic_half(p0: f32, p1: f32, p2: f32, p3: f32) -> f32 {
    let t = 0.5_f32;
    let a = 2.0 * p1;
    let b = -p0 + p2;
    let c = 2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3;
    let d = -p0 + 3.0 * p1 - 3.0 * p2 + p3;
    0.5 * (a + b * t + c * t * t + d * t * t * t)
}

fn densify_coords(values: &[f64]) -> Vec<f64> {
    if values.len() <= 1 {
        return values.to_vec();
    }

    let mut out = Vec::with_capacity(values.len() * 2 - 1);
    for window in values.windows(2) {
        let left = window[0];
        let right = window[1];
        out.push(left);
        out.push((left + right) * 0.5);
    }
    out.push(*values.last().expect("checked length"));
    out
}

fn clamp_chunk_shape(chunk_shape: [usize; 3], shape: [usize; 3]) -> [usize; 3] {
    [
        chunk_shape[0].max(1).min(shape[0].max(1)),
        chunk_shape[1].max(1).min(shape[1].max(1)),
        chunk_shape[2].max(1).min(shape[2].max(1)),
    ]
}
