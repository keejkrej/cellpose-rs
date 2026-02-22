use cellpose_rs::{preprocess, CellposeSession, SegmentParams};
use image::{GenericImageView, ImageReader};
use std::env;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: cargo run --example infer <image_path> <model_dir>");
        eprintln!("  e.g. cargo run --example infer cells.png models/cellpose-cpsam");
        std::process::exit(1);
    }

    let img_path = &args[1];
    let model_dir = Path::new(&args[2]);
    let model_path = model_dir.join("model.onnx");

    if !model_path.exists() {
        eprintln!("Model not found at: {}", model_path.display());
        std::process::exit(1);
    }

    println!("Loading image: {}", img_path);
    let img = ImageReader::open(img_path)?.decode()?;
    let (w, h) = img.dimensions();
    let (w, h) = (w as usize, h as usize);
    let img_rgb = img.to_rgb8();

    // Extract channels — adapt these to your image:
    //   Here we use green as fluorescence, blue as phase proxy.
    let mut phase = vec![0.0f32; w * h];
    let mut fluo = vec![0.0f32; w * h];

    for (x, y, pixel) in img_rgb.enumerate_pixels() {
        let idx = y as usize * w + x as usize;
        phase[idx] = pixel[2] as f32; // blue
        fluo[idx] = pixel[1] as f32;  // green
    }

    let chw = preprocess::build_chw_image(phase, fluo, h, w);

    println!("Loading ONNX session...");
    let mut session = CellposeSession::new(&model_path, false)?;

    println!("Running segmentation ({w}×{h}, {} tiles)...", {
        let hpad = ((h + 255) / 256) * 256;
        let wpad = ((w + 255) / 256) * 256;
        (hpad / 256) * (wpad / 256)
    });
    let t0 = Instant::now();
    let masks = session.segment(&chw, h, w, SegmentParams::default())?;
    println!("Total time: {:?}", t0.elapsed());

    let max_label = *masks.iter().max().unwrap_or(&0);
    println!("Found {} cells.", max_label);

    // Colorize and save
    let stem = Path::new(img_path).file_stem().unwrap().to_str().unwrap();
    let parent = Path::new(img_path).parent().unwrap_or(Path::new("."));
    let out_path = parent.join(format!("{}_masks.png", stem));
    let mut out_img = image::RgbImage::new(w as u32, h as u32);

    fn label_to_color(label: u32) -> [u8; 3] {
        if label == 0 { return [0, 0, 0]; }
        let hue = ((label as f32) * 137.5).rem_euclid(360.0);
        let c = 1.0f32;
        let x = c * (1.0 - ((hue / 60.0) % 2.0 - 1.0).abs());
        let (r, g, b) = match (hue as i32) / 60 {
            0 => (c, x, 0.0),
            1 => (x, c, 0.0),
            2 => (0.0, c, x),
            3 => (0.0, x, c),
            4 => (x, 0.0, c),
            _ => (c, 0.0, x),
        };
        [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
    }

    for y in 0..h {
        for x in 0..w {
            let color = label_to_color(masks[y * w + x]);
            out_img.put_pixel(x as u32, y as u32, image::Rgb(color));
        }
    }

    out_img.save(&out_path)?;
    println!("Saved: {}", out_path.display());
    Ok(())
}
