use std::fs;
use ultrahdr_rs::container;

fn zenjpeg_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(
        std::env::var("ZENJPEG_DIR").unwrap_or_else(|_| "/home/lilith/work/zenjpeg".into()),
    )
}

fn main() {
    let path = zenjpeg_dir().join("zenjpeg/tests/images/ultrahdr_sample.jpg");
    let data = fs::read(&path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}. Set ZENJPEG_DIR.", path.display(), e));

    // Parse using ultrahdr container functions
    let segments = container::scan_segments(&data);
    println!("Found {} segments", segments.len());

    // Find MPF segment
    for (i, seg) in segments.iter().enumerate() {
        println!(
            "Segment {}: marker={:02X}, offset={}, len={}",
            i,
            seg.marker_num,
            seg.offset,
            seg.data.len()
        );
        if seg.is_mpf() {
            println!("  -> MPF segment found!");
            match container::parse_mpf_segment(&seg.data, seg.offset) {
                Ok(mpf) => {
                    println!("  MPF parsed successfully:");
                    println!("    mpf_marker_offset: {}", mpf.mpf_marker_offset);
                    for (j, entry) in mpf.entries.iter().enumerate() {
                        println!(
                            "    Entry {}: type={:?}, size={}, offset={}",
                            j, entry.image_type, entry.size, entry.offset
                        );
                    }

                    // Try to extract secondary images
                    let secondaries = container::extract_secondary_images(&data, &mpf);
                    println!("  Found {} secondary images", secondaries.len());
                    for (k, sec) in secondaries.iter().enumerate() {
                        let preview: &[u8] = &sec[..2.min(sec.len())];
                        println!(
                            "    Secondary {}: {} bytes, starts with {:02X?}",
                            k,
                            sec.len(),
                            preview
                        );
                    }
                }
                Err(e) => {
                    println!("  MPF parse error: {:?}", e);
                }
            }
        }
    }
}
