# Ultra HDR Testing Plan

## Current State

The library has 47 unit tests embedded in source files, but **zero parity testing against the C++ libultrahdr reference implementation**. This document outlines what's needed for comprehensive validation.

## Priority: Critical Gaps

| Gap | Risk | Priority |
|-----|------|----------|
| No decode compatibility tests | Users can't open real Ultra HDR files | P0 |
| No encode parity tests | Output may not work with other decoders | P0 |
| No quality validation | Gain map may produce artifacts | P1 |
| No metadata compliance tests | XMP/ISO format may be wrong | P1 |
| No edge case coverage | Crashes on unusual inputs | P2 |

---

## Phase 1: Test Data Acquisition

### 1.1 Sample Ultra HDR Images

**Sources:**
- Google's libultrahdr test images: `https://github.com/google/libultrahdr/tree/main/tests/data`
- Android device captures (Pixel phones produce Ultra HDR)
- Adobe Lightroom exports (supports Ultra HDR)
- HDR+ enhanced photos from Google Photos

**Required samples:**
```
tests/data/
├── reference/                    # Known-good Ultra HDR JPEGs
│   ├── pixel_photo_01.jpg       # Real device capture
│   ├── pixel_photo_02.jpg
│   ├── lightroom_export.jpg     # Adobe workflow
│   └── libultrahdr_test_*.jpg   # From C++ test suite
├── sdr_sources/                  # SDR inputs for encoding tests
│   ├── kodak/                   # Kodak test set (24 images)
│   └── synthetic/               # Gradients, patterns, edge cases
├── hdr_sources/                  # HDR inputs for encoding tests
│   ├── openexr/                 # EXR files
│   ├── hdr_radiance/            # .hdr files
│   └── pq_heif/                 # PQ-encoded HEIF/AVIF
└── expected_output/              # C++ libultrahdr output for comparison
    ├── gainmaps/                # Extracted gain maps
    └── metadata/                # Serialized XMP/ISO metadata
```

**Action items:**
- [ ] Clone libultrahdr and extract test data
- [ ] Capture 10+ Ultra HDR photos from Pixel device
- [ ] Create synthetic test images (gradients, solid colors, patterns)
- [ ] Generate reference outputs using C++ libultrahdr

### 1.2 Reference Output Generation

Build libultrahdr and generate expected outputs:

```bash
# Clone and build libultrahdr
git clone https://github.com/google/libultrahdr.git
cd libultrahdr
mkdir build && cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..
ninja

# Generate reference encode outputs
./ultrahdr_app -m 0 -p hdr.p010 -w 1920 -h 1080 -o reference.jpg

# Extract gain map from existing Ultra HDR
./ultrahdr_app -m 1 -i ultrahdr.jpg -g gainmap.jpg -f metadata.txt
```

---

## Phase 2: Decode Compatibility Tests

### 2.1 Basic Decode Tests

```rust
// tests/decode_compatibility.rs

#[test]
fn test_decode_pixel_ultrahdr() {
    let data = include_bytes!("data/reference/pixel_photo_01.jpg");
    let decoder = Decoder::new(data).unwrap();

    assert!(decoder.is_ultrahdr());
    assert!(decoder.metadata().is_some());

    let sdr = decoder.decode_sdr().unwrap();
    assert!(sdr.width > 0 && sdr.height > 0);

    let hdr = decoder.decode_hdr(4.0).unwrap();
    assert_eq!(hdr.width, sdr.width);
    assert_eq!(hdr.height, sdr.height);
}

#[test]
fn test_decode_all_reference_images() {
    for entry in std::fs::read_dir("tests/data/reference").unwrap() {
        let path = entry.unwrap().path();
        if path.extension() == Some("jpg".as_ref()) {
            let data = std::fs::read(&path).unwrap();
            let result = Decoder::new(&data);
            assert!(result.is_ok(), "Failed to decode: {:?}", path);
        }
    }
}
```

### 2.2 Metadata Extraction Validation

Compare extracted metadata against C++ reference:

```rust
#[test]
fn test_metadata_extraction_parity() {
    let data = include_bytes!("data/reference/pixel_photo_01.jpg");
    let decoder = Decoder::new(data).unwrap();
    let metadata = decoder.metadata().unwrap();

    // Load expected values from C++ extraction
    let expected = load_expected_metadata("tests/data/expected_output/metadata/pixel_photo_01.json");

    assert_float_eq!(metadata.max_content_boost[0], expected.max_content_boost, 0.001);
    assert_float_eq!(metadata.hdr_capacity_max, expected.hdr_capacity_max, 0.001);
    assert_float_eq!(metadata.gamma[0], expected.gamma, 0.001);
}
```

### 2.3 Gain Map Extraction Validation

```rust
#[test]
fn test_gainmap_extraction_parity() {
    let data = include_bytes!("data/reference/pixel_photo_01.jpg");
    let decoder = Decoder::new(data).unwrap();
    let gainmap = decoder.decode_gainmap().unwrap();

    // Load C++ extracted gain map
    let expected_gm = load_grayscale_jpeg("tests/data/expected_output/gainmaps/pixel_photo_01_gm.jpg");

    assert_eq!(gainmap.width, expected_gm.width);
    assert_eq!(gainmap.height, expected_gm.height);

    // Compare pixel values (allow small tolerance for JPEG compression)
    let max_diff = compare_grayscale(&gainmap.data, &expected_gm.data);
    assert!(max_diff <= 2, "Gain map extraction differs by {} levels", max_diff);
}
```

---

## Phase 3: Encode Parity Tests

### 3.1 FFI Bindings to libultrahdr

Create `libultrahdr-sys` crate for direct comparison:

```rust
// internal/libultrahdr-sys/src/lib.rs

#[repr(C)]
pub struct uhdr_raw_image_t {
    pub fmt: uhdr_img_fmt_t,
    pub cg: uhdr_color_gamut_t,
    pub ct: uhdr_color_transfer_t,
    pub range: uhdr_color_range_t,
    pub w: u32,
    pub h: u32,
    pub planes: [*mut c_void; 3],
    pub stride: [u32; 3],
}

extern "C" {
    pub fn uhdr_create_encoder() -> *mut uhdr_codec_private_t;
    pub fn uhdr_enc_set_raw_image(
        enc: *mut uhdr_codec_private_t,
        img: *mut uhdr_raw_image_t,
        intent: uhdr_img_label_t,
    ) -> uhdr_error_info_t;
    pub fn uhdr_encode(enc: *mut uhdr_codec_private_t) -> uhdr_error_info_t;
    pub fn uhdr_get_encoded_stream(enc: *mut uhdr_codec_private_t) -> *mut uhdr_compressed_image_t;
    pub fn uhdr_release_encoder(enc: *mut uhdr_codec_private_t);
}
```

### 3.2 Byte-Level Comparison Tests

```rust
// tests/encode_parity.rs

#[test]
fn test_encode_parity_synthetic_gradient() {
    let hdr = create_hdr_gradient(256, 256);
    let sdr = create_sdr_gradient(256, 256);

    // Encode with Rust
    let rust_output = Encoder::new()
        .set_hdr_image(hdr.clone())
        .set_sdr_image(sdr.clone())
        .set_quality(90, 85)
        .encode()
        .unwrap();

    // Encode with C++ FFI
    let cpp_output = unsafe { encode_with_libultrahdr(&hdr, &sdr, 90, 85) };

    // Compare file sizes (should be within 5%)
    let size_ratio = rust_output.len() as f64 / cpp_output.len() as f64;
    assert!(size_ratio > 0.95 && size_ratio < 1.05,
        "Size mismatch: Rust {} vs C++ {}", rust_output.len(), cpp_output.len());
}
```

### 3.3 Quality Metric Comparison

```rust
#[test]
fn test_encode_quality_parity() {
    let hdr = load_hdr_image("tests/data/hdr_sources/openexr/memorial.exr");
    let sdr = tonemap_to_sdr(&hdr);

    // Encode with both implementations
    let rust_jpg = encode_rust(&hdr, &sdr);
    let cpp_jpg = encode_cpp(&hdr, &sdr);

    // Decode both and compare HDR reconstruction
    let rust_hdr = decode_to_hdr(&rust_jpg, 4.0);
    let cpp_hdr = decode_to_hdr(&cpp_jpg, 4.0);

    // Compare against original HDR using SSIMULACRA2
    let rust_ssim2 = compute_ssimulacra2(&hdr, &rust_hdr);
    let cpp_ssim2 = compute_ssimulacra2(&hdr, &cpp_hdr);

    // Rust should be within 2 points of C++
    assert!((rust_ssim2 - cpp_ssim2).abs() < 2.0,
        "Quality gap: Rust {:.1} vs C++ {:.1}", rust_ssim2, cpp_ssim2);
}
```

### 3.4 Gain Map Computation Parity

```rust
#[test]
fn test_gainmap_computation_parity() {
    let hdr = load_test_hdr();
    let sdr = load_test_sdr();

    // Compute gain map with both implementations
    let rust_gm = compute_gainmap_rust(&hdr, &sdr);
    let cpp_gm = compute_gainmap_cpp(&hdr, &sdr);

    // Compare dimensions
    assert_eq!(rust_gm.width, cpp_gm.width);
    assert_eq!(rust_gm.height, cpp_gm.height);

    // Compare values using DSSIM
    let dssim = compute_dssim_grayscale(&rust_gm.data, &cpp_gm.data, rust_gm.width, rust_gm.height);
    assert!(dssim < 0.001, "Gain map DSSIM: {}", dssim);
}
```

---

## Phase 4: Metadata Compliance Tests

### 4.1 XMP Serialization

```rust
#[test]
fn test_xmp_spec_compliance() {
    let metadata = GainMapMetadata {
        max_content_boost: [4.0; 3],
        min_content_boost: [1.0; 3],
        gamma: [1.0; 3],
        offset_sdr: [0.015625; 3],
        offset_hdr: [0.015625; 3],
        hdr_capacity_min: 1.0,
        hdr_capacity_max: 4.0,
        use_base_color_space: true,
    };

    let xmp = generate_xmp(&metadata, 10000);

    // Validate XMP structure
    assert!(xmp.contains("xmlns:hdrgm=\"http://ns.adobe.com/hdr-gain-map/1.0/\""));
    assert!(xmp.contains("hdrgm:Version=\"1.0\""));
    assert!(xmp.contains("hdrgm:GainMapMax="));

    // Parse back and verify round-trip
    let (parsed, _) = parse_xmp(&xmp).unwrap();
    assert_float_eq!(parsed.max_content_boost[0], metadata.max_content_boost[0], 0.0001);
}
```

### 4.2 ISO 21496-1 Binary Format

```rust
#[test]
fn test_iso21496_binary_format() {
    let metadata = create_test_metadata();

    // Serialize
    let binary = serialize_iso21496(&metadata);

    // Verify magic bytes and structure
    assert_eq!(&binary[0..4], b"????"); // Check actual magic

    // Parse with C++ implementation and compare
    let cpp_parsed = parse_iso21496_cpp(&binary);
    assert_eq!(cpp_parsed.max_content_boost, metadata.max_content_boost[0]);
}
```

### 4.3 MPF Structure Validation

```rust
#[test]
fn test_mpf_structure() {
    let encoded = encode_ultrahdr_test_image();

    // Parse MPF
    let images = parse_mpf(&encoded).unwrap();

    assert_eq!(images.len(), 2);

    // Verify primary image is valid JPEG
    let (primary_start, primary_end) = images[0];
    assert_eq!(&encoded[primary_start..primary_start+2], &[0xFF, 0xD8]);

    // Verify gain map is valid JPEG
    let (gm_start, gm_end) = images[1];
    assert_eq!(&encoded[gm_start..gm_start+2], &[0xFF, 0xD8]);
    assert_eq!(&encoded[gm_end-2..gm_end], &[0xFF, 0xD9]);
}
```

---

## Phase 5: Quality Validation

### 5.1 Tone Mapping Quality

```rust
#[test]
fn test_tonemap_quality() {
    for hdr_file in glob("tests/data/hdr_sources/**/*.exr") {
        let hdr = load_exr(&hdr_file);
        let sdr = tonemap_image_to_srgb8(&hdr, ColorGamut::Bt709);

        // SDR should preserve detail without obvious artifacts
        // Check that highlights aren't completely clipped
        let sdr_max: u8 = sdr.iter().copied().max().unwrap();
        assert!(sdr_max >= 250, "Highlights clipped in {:?}", hdr_file);

        // Check that shadows aren't crushed
        let sdr_min: u8 = sdr.iter().copied().min().unwrap();
        assert!(sdr_min <= 10, "Shadows crushed in {:?}", hdr_file);
    }
}
```

### 5.2 Round-Trip Quality

```rust
#[test]
fn test_roundtrip_quality() {
    let original_hdr = load_hdr_image("tests/data/hdr_sources/openexr/memorial.exr");

    // Encode to Ultra HDR
    let encoded = Encoder::new()
        .set_hdr_image(original_hdr.clone())
        .set_quality(95, 90)
        .set_target_display_peak(1000.0)
        .encode()
        .unwrap();

    // Decode back to HDR
    let decoder = Decoder::new(&encoded).unwrap();
    let reconstructed = decoder.decode_hdr(1000.0 / 203.0).unwrap();

    // Compare with SSIMULACRA2
    let ssim2 = compute_ssimulacra2(&original_hdr, &reconstructed);
    assert!(ssim2 > 70.0, "Round-trip quality too low: {}", ssim2);

    // Compare with Butteraugli
    let butteraugli = compute_butteraugli(&original_hdr, &reconstructed);
    assert!(butteraugli < 2.0, "Round-trip Butteraugli too high: {}", butteraugli);
}
```

### 5.3 Gain Map Application Accuracy

```rust
#[test]
fn test_gainmap_application_accuracy() {
    // Create known HDR/SDR pair
    let hdr = create_hdr_with_known_boost();  // e.g., SDR * 4.0 in highlights
    let sdr = create_sdr_base();

    // Compute and apply gain map
    let (gainmap, metadata) = compute_gainmap(&hdr, &sdr, &default_config());
    let reconstructed = apply_gainmap(&sdr, &gainmap, &metadata, 4.0, HdrOutputFormat::LinearFloat);

    // Verify reconstruction matches original HDR
    let max_error = compare_images_max_error(&hdr, &reconstructed);
    assert!(max_error < 0.02, "Gain map reconstruction error: {}", max_error);
}
```

---

## Phase 6: Edge Cases and Stress Tests

### 6.1 Dimension Edge Cases

```rust
#[test]
fn test_odd_dimensions() {
    for (w, h) in [(1, 1), (3, 3), (7, 7), (15, 15), (127, 127), (1, 1000), (1000, 1)] {
        let hdr = create_hdr_image(w, h);
        let result = Encoder::new().set_hdr_image(hdr).encode();
        assert!(result.is_ok(), "Failed for {}x{}", w, h);
    }
}

#[test]
fn test_non_mcu_aligned() {
    // Dimensions not divisible by 8 or 16
    for (w, h) in [(17, 17), (33, 65), (100, 101), (1920, 1080), (1919, 1079)] {
        let hdr = create_hdr_image(w, h);
        let encoded = Encoder::new().set_hdr_image(hdr).encode().unwrap();

        let decoder = Decoder::new(&encoded).unwrap();
        let (dec_w, dec_h) = decoder.dimensions().unwrap();
        assert_eq!((dec_w, dec_h), (w, h));
    }
}
```

### 6.2 Extreme Values

```rust
#[test]
fn test_extreme_hdr_values() {
    // Very bright highlights
    let mut hdr = create_hdr_image(256, 256);
    set_pixel_linear(&mut hdr, 128, 128, [100.0, 100.0, 100.0]);  // 100x SDR white

    let result = Encoder::new().set_hdr_image(hdr).encode();
    assert!(result.is_ok());
}

#[test]
fn test_zero_and_negative_values() {
    let mut hdr = create_hdr_image(256, 256);
    set_pixel_linear(&mut hdr, 0, 0, [0.0, 0.0, 0.0]);
    set_pixel_linear(&mut hdr, 1, 0, [-0.1, -0.1, -0.1]);  // Out of gamut

    let result = Encoder::new().set_hdr_image(hdr).encode();
    assert!(result.is_ok());
}
```

### 6.3 Quality Parameter Extremes

```rust
#[test]
fn test_quality_extremes() {
    let hdr = create_test_hdr();

    // Minimum quality
    let low_q = Encoder::new()
        .set_hdr_image(hdr.clone())
        .set_quality(1, 1)
        .encode()
        .unwrap();

    // Maximum quality
    let high_q = Encoder::new()
        .set_hdr_image(hdr.clone())
        .set_quality(100, 100)
        .encode()
        .unwrap();

    // High quality should be larger
    assert!(high_q.len() > low_q.len());
}
```

---

## Phase 7: Cross-Decoder Compatibility

### 7.1 Test with Multiple Decoders

```rust
#[test]
fn test_decode_with_libultrahdr() {
    let encoded = encode_rust_ultrahdr();

    // Decode with C++ libultrahdr
    let cpp_decoded = unsafe { decode_with_libultrahdr(&encoded) };
    assert!(cpp_decoded.is_ok(), "C++ decoder rejected our output");
}

#[test]
fn test_decode_with_android_framework() {
    // This would require Android emulator or device testing
    // Mark as ignored for CI, run manually
}
```

### 7.2 Adobe Lightroom Compatibility

```rust
#[test]
#[ignore]  // Manual test - requires Lightroom
fn test_lightroom_can_open_output() {
    let encoded = encode_test_image();
    std::fs::write("/tmp/test_ultrahdr.jpg", &encoded).unwrap();
    println!("Open /tmp/test_ultrahdr.jpg in Lightroom and verify HDR recognized");
}
```

---

## Implementation Checklist

### Test Infrastructure
- [ ] Create `tests/` directory structure
- [ ] Add `libultrahdr-sys` FFI bindings crate
- [ ] Set up test data download/generation scripts
- [ ] Add CI job for parity tests (may need large runner)

### Phase 1: Test Data
- [ ] Clone libultrahdr test images
- [ ] Capture Pixel Ultra HDR samples
- [ ] Generate synthetic test images
- [ ] Create expected output reference files

### Phase 2: Decode Tests
- [ ] Basic decode compatibility tests
- [ ] Metadata extraction validation
- [ ] Gain map extraction validation
- [ ] Error handling tests

### Phase 3: Encode Tests
- [ ] FFI bindings to libultrahdr
- [ ] File size comparison tests
- [ ] Quality metric comparison tests
- [ ] Gain map computation parity

### Phase 4: Metadata Tests
- [ ] XMP round-trip tests
- [ ] ISO 21496-1 format tests
- [ ] MPF structure tests

### Phase 5: Quality Tests
- [ ] Tone mapping quality validation
- [ ] Round-trip SSIMULACRA2 tests
- [ ] Butteraugli validation

### Phase 6: Edge Cases
- [ ] Odd dimension tests
- [ ] Extreme value tests
- [ ] Quality parameter tests

### Phase 7: Compatibility
- [ ] Cross-decoder tests
- [ ] Manual Lightroom test

---

## Success Criteria

| Metric | Target | Notes |
|--------|--------|-------|
| Decode compatibility | 100% | All reference Ultra HDR files must decode |
| Encode size parity | ±5% | vs C++ libultrahdr at same quality |
| Gain map DSSIM | <0.001 | vs C++ computation |
| Round-trip SSIM2 | >70 | Original HDR vs reconstructed |
| Round-trip Butteraugli | <2.0 | Original HDR vs reconstructed |
| Metadata round-trip | Exact | XMP and ISO 21496-1 |

---

## Resources

- libultrahdr source: https://github.com/google/libultrahdr
- Ultra HDR spec: https://developer.android.com/media/platform/hdr-image-format
- XMP namespace: http://ns.adobe.com/hdr-gain-map/1.0/
- ISO 21496-1: Gain map encoding standard
- CIPA DC-007: MPF specification
