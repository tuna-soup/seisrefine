use std::path::Path;

use ndarray::Array3;
use rayon::prelude::*;

use crate::error::SeisRefineError;
use crate::metadata::{DatasetKind, DerivedFrom, InterpMethod, StoreManifest, VolumeAxes};
use crate::store::{StoreHandle, create_store, load_array, open_store};

#[derive(Debug, Clone)]
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
    let output_array = upscale_linear_2x(&input_array);
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

pub fn upscale_linear_2x(input: &Array3<f32>) -> Array3<f32> {
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
                let left_i = out_i / 2;
                let left_x = out_x / 2;
                let row_offset = out_x * samples;

                let top_left = trace_slice(src, xlines, samples, left_i, left_x);
                let top_right = if out_x % 2 == 1 {
                    trace_slice(src, xlines, samples, left_i, left_x + 1)
                } else {
                    top_left
                };
                let bottom_left = if out_i % 2 == 1 {
                    trace_slice(src, xlines, samples, left_i + 1, left_x)
                } else {
                    top_left
                };
                let bottom_right = match (out_i % 2, out_x % 2) {
                    (1, 1) => trace_slice(src, xlines, samples, left_i + 1, left_x + 1),
                    (1, 0) => bottom_left,
                    (0, 1) => top_right,
                    _ => top_left,
                };

                let dst = &mut row[row_offset..row_offset + samples];
                match (out_i % 2, out_x % 2) {
                    (0, 0) => dst.copy_from_slice(top_left),
                    (0, 1) => average_pair(dst, top_left, top_right),
                    (1, 0) => average_pair(dst, top_left, bottom_left),
                    (1, 1) => average_quad(dst, top_left, top_right, bottom_left, bottom_right),
                    _ => unreachable!(),
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

fn average_pair(dst: &mut [f32], a: &[f32], b: &[f32]) {
    for ((dst, a), b) in dst.iter_mut().zip(a.iter().copied()).zip(b.iter().copied()) {
        *dst = (a + b) * 0.5;
    }
}

fn average_quad(dst: &mut [f32], a: &[f32], b: &[f32], c: &[f32], d: &[f32]) {
    for ((((dst, a), b), c), d) in dst
        .iter_mut()
        .zip(a.iter().copied())
        .zip(b.iter().copied())
        .zip(c.iter().copied())
        .zip(d.iter().copied())
    {
        *dst = (a + b + c + d) * 0.25;
    }
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
