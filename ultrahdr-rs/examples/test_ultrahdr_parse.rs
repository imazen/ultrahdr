use std::fs;
use ultrahdr_rs::container;

fn zenjpeg_dir() -> std::path::PathBuf {
    let dir = std::path::PathBuf::from(std::env::var("ZENJPEG_DIR").unwrap_or_else(|_| {
        // Default: sibling directory relative to workspace root
        let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(2)
            .expect("workspace root");
        workspace.join("zenjpeg").to_string_lossy().into_owned()
    }));
    assert!(
        dir.is_dir(),
        "zenjpeg repo not found: {}. Set ZENJPEG_DIR.",
        dir.display()
    );
    dir
}

fn main() {
    let path = zenjpeg_dir().join("zenjpeg/tests/images/ultrahdr_sample.jpg");
    let data = fs::read(&path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}. Set ZENJPEG_DIR.", path.display(), e));

    // APP segments overview.
    let segments = container::scan_segments(&data);
    println!("Found {} APP segments", segments.len());
    for (i, seg) in segments.iter().enumerate() {
        println!(
            "  [{i}] APP{}, offset={}, payload_len={}, mpf={}, xmp={}, icc={}, exif={}",
            seg.marker_num,
            seg.offset,
            seg.data.len(),
            seg.is_mpf(),
            seg.is_xmp(),
            seg.is_icc(),
            seg.is_exif(),
        );
    }

    // MPF directory.
    match container::parse_mpf(&data) {
        Ok(entries) if entries.is_empty() => println!("No MPF segment present."),
        Ok(entries) => {
            println!("\nMPF directory: {} entries", entries.len());
            for (i, e) in entries.iter().enumerate() {
                println!(
                    "  [{i}] type={:?}, offset={}, size={}",
                    e.image_type, e.offset, e.size
                );
            }
            // Show the first 2 bytes of each secondary image as a sanity peek.
            for (i, e) in entries.iter().enumerate().skip(1) {
                let end = e.offset + e.size;
                if end <= data.len() {
                    let preview = &data[e.offset..e.offset + 2.min(e.size)];
                    println!("  secondary {i}: starts with {:02X?}", preview);
                }
            }
        }
        Err(err) => println!("MPF parse failed: {err}"),
    }
}
