//! Cellpose cpsam inference in Rust via ONNX Runtime.
//!
//! Provides preprocessing, ONNX inference, and Cellpose flow-dynamics
//! post-processing to produce integer segmentation masks from microscopy images.
//!
//! # Example
//! ```no_run
//! use cellpose_rs::{preprocess, dynamics, CellposeSession};
//! use std::path::Path;
//!
//! let mut session = CellposeSession::new(Path::new("models/cellpose-cpsam/model.onnx"), false).unwrap();
//!
//! // Build a (3, H, W) CHW float32 image from two channels
//! let phase = vec![0.0f32; 256 * 256];
//! let fluo = vec![0.0f32; 256 * 256];
//! let chw = preprocess::build_chw_image(phase, fluo, 256, 256);
//!
//! // Run full pipeline: preprocess → infer → dynamics → masks
//! let masks = session.segment(&chw, 256, 256, Default::default()).unwrap();
//! ```

pub mod dynamics;
pub mod preprocess;

use ndarray::{Array, Ix4};
use ort::session::Session;
use ort::value::Tensor;
#[cfg(any(windows, target_os = "linux"))]
use ort::ep::{CUDA, ExecutionProvider};
use std::io::Write;
use std::path::Path;

/// Parameters for the segmentation pipeline.
pub struct SegmentParams {
    /// Tile size in pixels (must match the exported ONNX model). Default: 256.
    pub tile: usize,
    /// Batch size for ONNX inference. Default: 1.
    pub batch_size: usize,
    /// Cell probability threshold. Default: 0.0.
    pub cellprob_threshold: f32,
    /// Number of Euler integration steps. Default: 200.
    pub niter: usize,
    /// Minimum mask size in pixels. Default: 15.
    pub min_size: usize,
}

impl Default for SegmentParams {
    fn default() -> Self {
        Self {
            tile: 256,
            batch_size: 1,
            cellprob_threshold: 0.0,
            niter: 200,
            min_size: 15,
        }
    }
}

/// Wraps an ONNX Runtime session for Cellpose cpsam inference.
pub struct CellposeSession {
    session: Session,
    input_name: String,
}

impl CellposeSession {
    /// Create a new session from a model.onnx path.
    /// Set `cpu = true` to force CPU execution (skip CUDA).
    pub fn new(model_path: &Path, cpu: bool) -> Result<Self, Box<dyn std::error::Error>> {
        let mut builder = Session::builder()?;
        #[cfg(any(windows, target_os = "linux"))]
        if !cpu {
            let cuda = CUDA::default();
            match cuda.is_available() {
                Ok(true) => {
                    if let Ok(()) = cuda.register(&mut builder) {
                        eprintln!("cellpose-rs: using CUDA.");
                        let _ = std::io::stderr().flush();
                    }
                }
                _ => {}
            }
        }
        let session = builder.commit_from_file(model_path)?;
        let input_name = session.inputs().first().ok_or("No inputs")?.name().to_string();
        Ok(Self { session, input_name })
    }

    /// Run inference on a batch of tiles.
    /// Each tile is a flat `Vec<f32>` of length `3 * tile * tile` in CHW order.
    /// Returns one output vec per tile, also `3 * tile * tile` in CHW order: `[dY, dX, cellprob]`.
    pub fn infer_tiles(
        &mut self,
        tiles: &[Vec<f32>],
        tile: usize,
        batch_size: usize,
    ) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        let n = tiles.len();
        let mut outputs = Vec::with_capacity(n);

        for chunk in tiles.chunks(batch_size) {
            let bs = chunk.len();
            let mut data = vec![0.0f32; bs * 3 * tile * tile];
            for (i, t) in chunk.iter().enumerate() {
                let off = i * 3 * tile * tile;
                data[off..off + 3 * tile * tile].copy_from_slice(t);
            }
            let shape: Ix4 = ndarray::Dim([bs, 3, tile, tile]);
            let arr = Array::from_shape_vec(shape, data)?;
            let input_tensor = Tensor::from_array(arr)?;
            let inputs = ort::inputs![self.input_name.as_str() => input_tensor];
            let result = self.session.run(inputs)?;
            let out_arr = result[0].try_extract_array::<f32>()?;
            let out_flat: Vec<f32> = out_arr.iter().cloned().collect();
            for i in 0..bs {
                let off = i * 3 * tile * tile;
                outputs.push(out_flat[off..off + 3 * tile * tile].to_vec());
            }
        }
        Ok(outputs)
    }

    /// Full segmentation pipeline: pad → tile → infer → stitch → dynamics → masks.
    ///
    /// `chw` is a flat `(3, H, W)` CHW float32 image (already percentile-normalised).
    /// Returns a `Vec<u32>` of length `H * W` with integer cell labels (0 = background).
    pub fn segment(
        &mut self,
        chw: &[f32],
        h: usize,
        w: usize,
        params: SegmentParams,
    ) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let tile = params.tile;

        // Pad + tile
        let (padded, hpad, wpad) = preprocess::pad_chw(chw, h, w, tile);
        let (tiles, ny, nx) = preprocess::extract_tiles(&padded, hpad, wpad, tile);

        // Infer
        let tile_outs = self.infer_tiles(&tiles, tile, params.batch_size)?;

        // Stitch
        let stitched = preprocess::stitch_tiles(&tile_outs, ny, nx, h, w, tile);

        // Post-process
        let masks = dynamics::flows_to_masks(
            &stitched, h, w,
            params.cellprob_threshold,
            params.niter,
            params.min_size,
        );

        Ok(masks)
    }
}
