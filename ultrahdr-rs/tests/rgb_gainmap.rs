//! Issue #27 regression: RGB (multi-channel) gain maps must decode.
//!
//! `decode_gainmap` used to request grayscale output unconditionally —
//! failing outright for some color encodings ("unsupported color
//! conversion", e.g. the libavif seine sample whose hdrgm metadata carries
//! distinct per-channel triples) and silently luma-averaging any RGB map
//! that did decode. It now decodes RGB and collapses to single-channel
//! only when the map is provably achromatic.

use ultrahdr_core::{ColorPrimaries, GainMapMetadata, PixelFormat, TransferFunction,
    pixel_buffer_from_vec};
use ultrahdr_rs::{Decoder, Encoder};
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, Unstoppable};

/// Mint a small JPEG with the given layout and pixel bytes.
fn mint_jpeg(w: u32, h: u32, layout: PixelLayout, px: &[u8], row_bytes: usize) -> Vec<u8> {
    // 4:4:4 so per-channel differences survive into the decoded map.
    let cfg = EncoderConfig::ycbcr(92.0, ChromaSubsampling::None);
    let mut enc = cfg.encode_from_bytes(w, h, layout).unwrap();
    enc.push(px, h as usize, row_bytes, Unstoppable).unwrap();
    enc.finish().unwrap()
}

/// A colorful SDR base (content irrelevant — the gain map is what's tested).
fn sdr_base(w: u32, h: u32) -> ultrahdr_core::PixelBuffer {
    let mut data = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            data.push((x * 255 / w.max(1)) as u8);
            data.push((y * 255 / h.max(1)) as u8);
            data.push(128);
        }
    }
    pixel_buffer_from_vec(
        data,
        w,
        h,
        PixelFormat::Rgb8,
        ColorPrimaries::Bt709,
        TransferFunction::Srgb,
    )
    .unwrap()
}

/// Per-channel ISO 21496-1 metadata with distinct channel maxes.
fn rgb_metadata() -> GainMapMetadata {
    let mut m = GainMapMetadata::default();
    for (i, max) in [1.2f64, 1.3, 1.1].into_iter().enumerate() {
        m.channels[i].min = 0.0;
        m.channels[i].max = max;
        m.channels[i].gamma = 1.0;
        m.channels[i].base_offset = 1.0 / 64.0;
        m.channels[i].alternate_offset = 1.0 / 64.0;
    }
    m.alternate_hdr_headroom = 1.3;
    m.use_base_color_space = true;
    m
}

#[test]
fn rgb_gainmap_decodes_three_channels() {
    // A 16×16 gain-map JPEG with real chroma (R/G/B planes differ).
    let (w, h) = (16u32, 16u32);
    let mut gm_px = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            gm_px.push((40 + x * 12) as u8);
            gm_px.push((200 - y * 11) as u8);
            gm_px.push(90);
        }
    }
    let gm_jpeg = mint_jpeg(w, h, PixelLayout::Rgb8Srgb, &gm_px, (w * 3) as usize);

    let mut encoder = Encoder::new();
    encoder
        .set_sdr_image(sdr_base(32, 32))
        .set_gainmap_jpeg(gm_jpeg, rgb_metadata());
    let bytes = encoder.encode().expect("encode UltraHDR with RGB gain map");

    let d = Decoder::new(&bytes).expect("decode container");
    assert!(d.is_ultrahdr());
    let gm = d.decode_gainmap().expect("RGB gain map must decode (#27)");
    assert_eq!((gm.width, gm.height), (w, h));
    assert_eq!(gm.channels, 3, "per-channel map must stay 3-channel");
    assert_eq!(gm.data.len(), (w * h * 3) as usize);
    // Chroma survived: at least one pixel with R != G.
    assert!(
        gm.data.chunks_exact(3).any(|px| px[0] != px[1]),
        "decoded map lost its chroma"
    );
}

#[test]
fn achromatic_gainmap_collapses_to_single_channel() {
    // The same path with a gray-coded map must keep the 1-channel form.
    let (w, h) = (16u32, 16u32);
    let gm_px: Vec<u8> = (0..w * h).map(|i| (i % 251) as u8).collect();
    let cfg = EncoderConfig::grayscale(92.0);
    let mut enc = cfg
        .encode_from_bytes(w, h, PixelLayout::Gray8Srgb)
        .unwrap();
    enc.push(&gm_px, h as usize, w as usize, Unstoppable).unwrap();
    let gm_jpeg = enc.finish().unwrap();

    let mut encoder = Encoder::new();
    encoder
        .set_sdr_image(sdr_base(32, 32))
        .set_gainmap_jpeg(gm_jpeg, rgb_metadata());
    let bytes = encoder.encode().expect("encode UltraHDR with gray gain map");

    let gm = Decoder::new(&bytes)
        .expect("decode container")
        .decode_gainmap()
        .expect("gray gain map must decode");
    assert_eq!(gm.channels, 1, "achromatic map must collapse to 1 channel");
    assert_eq!(gm.data.len(), (w * h) as usize);
}
