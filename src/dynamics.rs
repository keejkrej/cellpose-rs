//! Cellpose flow dynamics post-processing.
//!
//! Converts raw network output `(dY, dX, cellprob)` into integer segmentation masks
//! using Euler integration of gradient flows, histogram-based seed detection,
//! connected-component labelling, hole filling, and small mask removal.

use ndarray::{Array2, Array3};
use std::collections::HashMap;

/// Bilinear sample of a 2D flow field (2, H, W) at sub-pixel positions.
fn bilinear_sample_flows(
    flow: &Array3<f32>,
    pos_y: &[f32],
    pos_x: &[f32],
    h: usize,
    w: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n = pos_y.len();
    let mut dy_out = vec![0.0f32; n];
    let mut dx_out = vec![0.0f32; n];

    let hf = (h - 1) as f32;
    let wf = (w - 1) as f32;

    for i in 0..n {
        let y = pos_y[i].clamp(0.0, hf);
        let x = pos_x[i].clamp(0.0, wf);
        let y0 = y.floor() as usize;
        let x0 = x.floor() as usize;
        let y1 = (y0 + 1).min(h - 1);
        let x1 = (x0 + 1).min(w - 1);
        let fy = y - y0 as f32;
        let fx = x - x0 as f32;

        let interp = |ch: usize| -> f32 {
            let f00 = flow[[ch, y0, x0]];
            let f10 = flow[[ch, y1, x0]];
            let f01 = flow[[ch, y0, x1]];
            let f11 = flow[[ch, y1, x1]];
            f00 * (1.0 - fy) * (1.0 - fx)
                + f10 * fy * (1.0 - fx)
                + f01 * (1.0 - fy) * fx
                + f11 * fy * fx
        };
        dy_out[i] = interp(0);
        dx_out[i] = interp(1);
    }
    (dy_out, dx_out)
}

/// Run Cellpose-style Euler flow dynamics on foreground pixels.
fn follow_flows(
    flow: &Array3<f32>,
    inds_y: &[usize],
    inds_x: &[usize],
    h: usize,
    w: usize,
    niter: usize,
) -> (Vec<i32>, Vec<i32>) {
    let n = inds_y.len();
    let mut pos_y: Vec<f32> = inds_y.iter().map(|&v| v as f32).collect();
    let mut pos_x: Vec<f32> = inds_x.iter().map(|&v| v as f32).collect();

    let hf = (h - 1) as f32;
    let wf = (w - 1) as f32;

    for _ in 0..niter {
        let (dy, dx) = bilinear_sample_flows(flow, &pos_y, &pos_x, h, w);
        for i in 0..n {
            pos_y[i] = (pos_y[i] + dy[i]).clamp(0.0, hf);
            pos_x[i] = (pos_x[i] + dx[i]).clamp(0.0, wf);
        }
    }

    let iy: Vec<i32> = pos_y.iter().map(|&v| v.round() as i32).collect();
    let ix: Vec<i32> = pos_x.iter().map(|&v| v.round() as i32).collect();
    (iy, ix)
}

/// Max-pool over a 2D histogram grid (kernel × kernel, stride=1, same padding).
fn max_pool_2d(h_grid: &Array2<u32>, kernel: usize) -> Array2<u32> {
    let (rows, cols) = (h_grid.nrows(), h_grid.ncols());
    let half = kernel / 2;
    let mut out = Array2::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            let r0 = r.saturating_sub(half);
            let r1 = (r + half + 1).min(rows);
            let c0 = c.saturating_sub(half);
            let c1 = (c + half + 1).min(cols);
            let mut mx = 0u32;
            for rr in r0..r1 {
                for cc in c0..c1 {
                    mx = mx.max(h_grid[[rr, cc]]);
                }
            }
            out[[r, c]] = mx;
        }
    }
    out
}

/// Connected-component labelling (4-connectivity) on a boolean mask.
/// Returns label array (0 = background).
fn label_components(mask: &[bool], h: usize, w: usize) -> Vec<u32> {
    let mut labels = vec![0u32; h * w];
    let mut next_label = 1u32;
    let mut parent: Vec<u32> = vec![0u32; h * w + 1];
    for i in 0..parent.len() {
        parent[i] = i as u32;
    }
    fn find(parent: &mut Vec<u32>, x: u32) -> u32 {
        let mut x = x;
        while parent[x as usize] != x {
            parent[x as usize] = parent[parent[x as usize] as usize];
            x = parent[x as usize];
        }
        x
    }
    fn union(parent: &mut Vec<u32>, a: u32, b: u32) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent[ra as usize] = rb;
        }
    }
    for r in 0..h {
        for c in 0..w {
            let idx = r * w + c;
            if !mask[idx] {
                continue;
            }
            let top = if r > 0 { labels[(r - 1) * w + c] } else { 0 };
            let left = if c > 0 { labels[r * w + c - 1] } else { 0 };
            match (top > 0, left > 0) {
                (false, false) => {
                    labels[idx] = next_label;
                    parent[next_label as usize] = next_label;
                    next_label += 1;
                }
                (true, false) => labels[idx] = top,
                (false, true) => labels[idx] = left,
                (true, true) => {
                    labels[idx] = top;
                    union(&mut parent, top, left);
                }
            }
        }
    }
    for v in labels.iter_mut() {
        if *v > 0 {
            *v = find(&mut parent, *v);
        }
    }
    let mut remap = HashMap::<u32, u32>::new();
    let mut counter = 0u32;
    for v in labels.iter_mut() {
        if *v > 0 {
            let e = remap.entry(*v).or_insert_with(|| { counter += 1; counter });
            *v = *e;
        }
    }
    labels
}

/// Fill holes in each individual mask label.
fn fill_holes(masks: &mut Vec<u32>, h: usize, w: usize) {
    let max_label = *masks.iter().max().unwrap_or(&0);
    if max_label == 0 {
        return;
    }
    let mut bb_min_r = vec![h; max_label as usize + 1];
    let mut bb_max_r = vec![0usize; max_label as usize + 1];
    let mut bb_min_c = vec![w; max_label as usize + 1];
    let mut bb_max_c = vec![0usize; max_label as usize + 1];
    for r in 0..h {
        for c in 0..w {
            let lbl = masks[r * w + c] as usize;
            if lbl == 0 { continue; }
            bb_min_r[lbl] = bb_min_r[lbl].min(r);
            bb_max_r[lbl] = bb_max_r[lbl].max(r);
            bb_min_c[lbl] = bb_min_c[lbl].min(c);
            bb_max_c[lbl] = bb_max_c[lbl].max(c);
        }
    }
    for lbl in 1..=max_label as usize {
        let r0 = bb_min_r[lbl]; let r1 = bb_max_r[lbl];
        let c0 = bb_min_c[lbl]; let c1 = bb_max_c[lbl];
        if r0 > r1 || c0 > c1 { continue; }
        let bh = r1 - r0 + 1;
        let bw = c1 - c0 + 1;
        let mut crop_bg = vec![true; bh * bw];
        for r in r0..=r1 {
            for c in c0..=c1 {
                if masks[r * w + c] as usize == lbl {
                    crop_bg[(r - r0) * bw + (c - c0)] = false;
                }
            }
        }
        let mut visited = vec![false; bh * bw];
        let mut queue = std::collections::VecDeque::new();
        for r in 0..bh {
            for c in 0..bw {
                if (r == 0 || r == bh - 1 || c == 0 || c == bw - 1) && crop_bg[r * bw + c] {
                    visited[r * bw + c] = true;
                    queue.push_back((r, c));
                }
            }
        }
        while let Some((r, c)) = queue.pop_front() {
            for (dr, dc) in [(-1i32, 0), (1, 0), (0, -1i32), (0, 1)] {
                let nr = r as i32 + dr;
                let nc = c as i32 + dc;
                if nr < 0 || nr >= bh as i32 || nc < 0 || nc >= bw as i32 { continue; }
                let (nr, nc) = (nr as usize, nc as usize);
                if !visited[nr * bw + nc] && crop_bg[nr * bw + nc] {
                    visited[nr * bw + nc] = true;
                    queue.push_back((nr, nc));
                }
            }
        }
        for r in 0..bh {
            for c in 0..bw {
                if crop_bg[r * bw + c] && !visited[r * bw + c] {
                    masks[(r + r0) * w + (c + c0)] = lbl as u32;
                }
            }
        }
    }
}

/// Remove masks smaller than `min_size` pixels; renumber remaining labels 1..N.
fn remove_small_masks_and_renumber(masks: &mut Vec<u32>, min_size: usize) {
    let max_label = *masks.iter().max().unwrap_or(&0);
    if max_label == 0 { return; }
    let mut counts = vec![0usize; max_label as usize + 1];
    for &v in masks.iter() { counts[v as usize] += 1; }
    let mut remap = vec![0u32; max_label as usize + 1];
    let mut counter = 0u32;
    for lbl in 1..=max_label as usize {
        if counts[lbl] >= min_size {
            counter += 1;
            remap[lbl] = counter;
        }
    }
    for v in masks.iter_mut() {
        *v = remap[*v as usize];
    }
}

/// Convert Cellpose network output (3, H, W) into a u32 mask.
///
/// Input channels: `[dY, dX, cellprob]` in CHW order.
/// Returns a `Vec<u32>` of length `H * W` with integer cell labels (0 = background).
pub fn flows_to_masks(
    outputs: &[f32],
    h: usize,
    w: usize,
    cellprob_threshold: f32,
    niter: usize,
    min_size: usize,
) -> Vec<u32> {
    let dy_slice = &outputs[..h * w];
    let dx_slice = &outputs[h * w..2 * h * w];
    let cp_slice = &outputs[2 * h * w..];

    let mut inds_y = Vec::new();
    let mut inds_x = Vec::new();
    for r in 0..h {
        for c in 0..w {
            if cp_slice[r * w + c] > cellprob_threshold {
                inds_y.push(r);
                inds_x.push(c);
            }
        }
    }

    if inds_y.is_empty() {
        return vec![0u32; h * w];
    }

    let mut flow_dy = Array3::<f32>::zeros((2, h, w));
    {
        let mut fg_mask = vec![0.0f32; h * w];
        for (&iy, &ix) in inds_y.iter().zip(inds_x.iter()) {
            fg_mask[iy * w + ix] = 1.0;
        }
        for r in 0..h {
            for c in 0..w {
                let fg = fg_mask[r * w + c];
                flow_dy[[0, r, c]] = dy_slice[r * w + c] * fg / 5.0;
                flow_dy[[1, r, c]] = dx_slice[r * w + c] * fg / 5.0;
            }
        }
    }

    let (final_y, final_x) = follow_flows(&flow_dy, &inds_y, &inds_x, h, w, niter);

    let rpad = 20usize;
    let hg = h + 2 * rpad;
    let wg = w + 2 * rpad;
    let mut hist = Array2::<u32>::zeros((hg, wg));
    for i in 0..inds_y.len() {
        let ry = (final_y[i] + rpad as i32).clamp(0, hg as i32 - 1) as usize;
        let rx = (final_x[i] + rpad as i32).clamp(0, wg as i32 - 1) as usize;
        hist[[ry, rx]] += 1;
    }

    let hmax = max_pool_2d(&hist, 5);

    let mut seed_mask = vec![false; hg * wg];
    for r in 0..hg {
        for c in 0..wg {
            if hist[[r, c]] > 10 && (hist[[r, c]] as i32 - hmax[[r, c]] as i32) >= -1 {
                seed_mask[r * wg + c] = true;
            }
        }
    }

    if !seed_mask.iter().any(|&v| v) {
        return vec![0u32; h * w];
    }

    let seed_labels = label_components(&seed_mask, hg, wg);
    let n_seeds = *seed_labels.iter().max().unwrap_or(&0) as usize;
    if n_seeds == 0 {
        return vec![0u32; h * w];
    }

    let mut masks = vec![0u32; h * w];
    for i in 0..inds_y.len() {
        let ry = (final_y[i] + rpad as i32).clamp(0, hg as i32 - 1) as usize;
        let rx = (final_x[i] + rpad as i32).clamp(0, wg as i32 - 1) as usize;
        let lbl = seed_labels[ry * wg + rx];
        masks[inds_y[i] * w + inds_x[i]] = lbl;
    }

    let max_size = (h * w * 40 / 100).max(1);
    let max_label = *masks.iter().max().unwrap_or(&0);
    let mut counts = vec![0usize; max_label as usize + 1];
    for &v in masks.iter() { counts[v as usize] += 1; }
    for v in masks.iter_mut() {
        if *v > 0 && counts[*v as usize] > max_size {
            *v = 0;
        }
    }

    fill_holes(&mut masks, h, w);
    remove_small_masks_and_renumber(&mut masks, min_size);

    masks
}
