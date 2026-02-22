//! Image preprocessing utilities for Cellpose inference.

/// Percentile-normalize a 2D slice in-place (1st / 99th pct → 0.0 / 1.0, clamp to [0,1]).
pub fn percentile_normalize(data: &mut [f32]) {
    if data.is_empty() {
        return;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let lo = sorted[(n / 100).max(0).min(n - 1)];
    let hi = sorted[(n * 99 / 100).min(n - 1)];
    let range = hi - lo;
    if range < 1e-10 {
        for v in data.iter_mut() {
            *v = 0.0;
        }
    } else {
        for v in data.iter_mut() {
            *v = ((*v - lo) / range).clamp(0.0, 1.0);
        }
    }
}

/// Build a float32 (3, H, W) CHW image from phase + fluo frames, percentile-normalised.
/// Channels: `[phase, fluo, phase]` (matching Cellpose convention).
pub fn build_chw_image(
    phase: Vec<f32>,
    fluo: Vec<f32>,
    h: usize,
    w: usize,
) -> Vec<f32> {
    let mut phase = phase;
    let mut fluo = fluo;
    percentile_normalize(&mut phase);
    percentile_normalize(&mut fluo);

    let mut out = vec![0.0f32; 3 * h * w];
    out[..h * w].copy_from_slice(&phase);
    out[h * w..2 * h * w].copy_from_slice(&fluo);
    out[2 * h * w..].copy_from_slice(&phase);
    out
}

/// Pad a (3, H, W) CHW buffer to (3, Hpad, Wpad) where Hpad/Wpad are multiples of `tile`.
pub fn pad_chw(data: &[f32], h: usize, w: usize, tile: usize) -> (Vec<f32>, usize, usize) {
    let hpad = ((h + tile - 1) / tile) * tile;
    let wpad = ((w + tile - 1) / tile) * tile;
    let mut out = vec![0.0f32; 3 * hpad * wpad];
    for c in 0..3 {
        for row in 0..h {
            let src_off = c * h * w + row * w;
            let dst_off = c * hpad * wpad + row * wpad;
            out[dst_off..dst_off + w].copy_from_slice(&data[src_off..src_off + w]);
        }
    }
    (out, hpad, wpad)
}

/// Extract all (3, tile, tile) tiles from padded (3, Hpad, Wpad) CHW image.
/// Returns `(tiles, n_tiles_y, n_tiles_x)`.
pub fn extract_tiles(padded: &[f32], hpad: usize, wpad: usize, tile: usize)
    -> (Vec<Vec<f32>>, usize, usize)
{
    let ny = hpad / tile;
    let nx = wpad / tile;
    let mut tiles = Vec::with_capacity(ny * nx);
    for ty in 0..ny {
        for tx in 0..nx {
            let mut t = vec![0.0f32; 3 * tile * tile];
            for c in 0..3 {
                for row in 0..tile {
                    let src_off = c * hpad * wpad + (ty * tile + row) * wpad + tx * tile;
                    let dst_off = c * tile * tile + row * tile;
                    t[dst_off..dst_off + tile].copy_from_slice(&padded[src_off..src_off + tile]);
                }
            }
            tiles.push(t);
        }
    }
    (tiles, ny, nx)
}

/// Stitch tile outputs back into a (3, H, W) array (cropped from padded dimensions).
pub fn stitch_tiles(
    tile_outputs: &[Vec<f32>],
    ny: usize,
    nx: usize,
    h: usize,
    w: usize,
    tile: usize,
) -> Vec<f32> {
    let hpad = ny * tile;
    let wpad = nx * tile;
    let mut full = vec![0.0f32; 3 * hpad * wpad];
    for (idx, t) in tile_outputs.iter().enumerate() {
        let ty = idx / nx;
        let tx = idx % nx;
        for c in 0..3 {
            for row in 0..tile {
                let src_off = c * tile * tile + row * tile;
                let dst_off = c * hpad * wpad + (ty * tile + row) * wpad + tx * tile;
                full[dst_off..dst_off + tile].copy_from_slice(&t[src_off..src_off + tile]);
            }
        }
    }
    let mut out = vec![0.0f32; 3 * h * w];
    for c in 0..3 {
        for row in 0..h {
            let src_off = c * hpad * wpad + row * wpad;
            let dst_off = c * h * w + row * w;
            out[dst_off..dst_off + w].copy_from_slice(&full[src_off..src_off + w]);
        }
    }
    out
}
