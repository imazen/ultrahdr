use ultrahdr_core::gainmap::apply::{apply_gainmap, apply_gainmap_slice, calculate_weight};
use ultrahdr_core::gainmap::streaming::{RowDecoder, StreamDecoder};
use ultrahdr_core::{
    ColorPrimaries, GainMap, GainMapChannel, GainMapMetadata, HdrOutputFormat, PixelBuffer,
    PixelFormat, PixelSlice, TransferFunction, Unstoppable, descriptor_for, pixel_buffer_from_vec,
};

const PRIMARIES: ColorPrimaries = ColorPrimaries::Bt709;

fn linear_buffer(pixels: &[[f32; 4]], width: u32, height: u32) -> PixelBuffer {
    pixel_buffer_from_vec(
        bytemuck::cast_slice(pixels).to_vec(),
        width,
        height,
        PixelFormat::RGBAF32,
        PRIMARIES,
        TransferFunction::Linear,
    )
    .unwrap()
}

fn constant_curve(stops: f64, alternate_offset: f64) -> GainMapChannel {
    GainMapChannel {
        min: stops,
        max: stops,
        gamma: 1.0,
        base_offset: 1.0 / 64.0,
        alternate_offset,
    }
}

fn assert_bits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a.to_bits(), e.to_bits(), "sample {i}: {a} versus {e}");
    }
}

fn assert_renderers(
    base: &[[f32; 4]],
    gm: &GainMap,
    meta: &GainMapMetadata,
    boost: f32,
    expected: &[[f32; 4]],
) {
    let w = base.len() as u32;
    let output = apply_gainmap(
        &linear_buffer(base, w, 1),
        gm,
        meta,
        boost,
        HdrOutputFormat::LinearFloat,
        Unstoppable,
    )
    .unwrap();
    let actual = output.as_view().row_as::<f32>(0).unwrap();
    assert_bits(actual, expected.as_flattened());
    let mut row = RowDecoder::new(gm.clone(), meta.clone(), w, 1, boost, PRIMARIES).unwrap();
    let actual = row.process_rgba_rows(base.as_flattened(), 1).unwrap();
    assert_bits(&actual, expected.as_flattened());
    let mut stream = StreamDecoder::new(
        meta.clone(),
        w,
        1,
        gm.width,
        gm.height,
        gm.channels,
        boost,
        PRIMARIES,
    )
    .unwrap();
    if calculate_weight(boost, meta) != 0.0 {
        stream.push_gainmap_rows(&gm.data, gm.height).unwrap();
    } else {
        assert!(stream.can_process(1));
    }
    let actual = stream
        .process_sdr_rgba_rows(base.as_flattened(), 1)
        .unwrap();
    assert_bits(&actual, expected.as_flattened());
}

#[test]
fn zero_weight_bypasses_offsets_and_preserves_straight_alpha() {
    let base = [
        [-0.0, 0.5, 0.75, 0.0],
        [0.25, 0.125, 0.625, 0.25],
        [0.375, 0.375, 0.0, 0.5],
        [0.5, 0.25, 0.125, 1.0],
    ];
    let gm = GainMap::new(4, 1).unwrap();
    let mut meta = GainMapMetadata::default();
    meta.alternate_hdr_headroom = 2.0;
    meta.channels = [constant_curve(2.0, 3.0 / 64.0); 3];
    assert_eq!(calculate_weight(1.0, &meta).to_bits(), 0);
    assert_renderers(&base, &gm, &meta, 1.0, &base);
    // At one stop between zero and two, the constant two-stop curve doubles.
    let expected = base.map(|[r, g, b, a]| {
        [
            2.0 * r - 1.0 / 64.0,
            2.0 * g - 1.0 / 64.0,
            2.0 * b - 1.0 / 64.0,
            a,
        ]
    });
    assert_renderers(&base, &gm, &meta, 2.0, &expected);

    let input = linear_buffer(&base, 4, 1);
    let srgb = apply_gainmap(&input, &gm, &meta, 1.0, HdrOutputFormat::Srgb8, Unstoppable).unwrap();
    let alpha: Vec<_> = srgb
        .as_view()
        .row(0)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|p| p[3])
        .collect();
    assert_eq!(alpha, [0, 64, 128, 255]);
    #[cfg(feature = "f16")]
    {
        let output = apply_gainmap(
            &input,
            &gm,
            &meta,
            1.0,
            HdrOutputFormat::LinearF16,
            Unstoppable,
        )
        .unwrap();
        let expected = base
            .as_flattened()
            .iter()
            .map(|&v| half::f16::from_f32(v).to_bits())
            .collect::<Vec<_>>();
        assert_eq!(output.as_view().row_as::<u16>(0).unwrap(), expected);
    }
}

#[test]
fn hdr_base_weights_and_reverse_pixels_are_analytic() {
    let base = [[0.5, 0.25, 0.75, 0.5]];
    let gm = GainMap::new(1, 1).unwrap();
    let mut meta = GainMapMetadata::default();
    meta.base_hdr_headroom = 3.0;
    meta.channels = [constant_curve(3.0, 3.0 / 64.0); 3];
    assert_eq!(calculate_weight(1.0, &meta), -1.0);
    assert_eq!(calculate_weight(2.0, &meta), -2.0 / 3.0);
    assert_eq!(calculate_weight(8.0, &meta), 0.0);
    assert_renderers(&base, &gm, &meta, 8.0, &base);
    // (base + 1/64) / 8 - 3/64, independently reduced over denominator 512.
    let expected = [[
        (33.0 - 24.0) / 512.0,
        (17.0 - 24.0) / 512.0,
        (49.0 - 24.0) / 512.0,
        0.5,
    ]];
    assert_renderers(&base, &gm, &meta, 1.0, &expected);
    meta.alternate_hdr_headroom = 3.0;
    assert_renderers(&base, &gm, &meta, 2.0, &base);
}

#[test]
fn jpeg_iso_numeric_record_arity_and_gain_sample_arity_are_independent() {
    let base = [[0.25, 0.5, 0.75, 0.5]];
    for channels in [1, 3] {
        let gm = GainMap {
            width: 1,
            height: 1,
            channels,
            data: if channels == 1 {
                vec![127]
            } else {
                vec![127, 0, 255]
            },
        };
        assert_eq!(gm.data.len(), usize::from(channels));
        for records in [1, 3] {
            let mut meta = GainMapMetadata::default();
            meta.alternate_hdr_headroom = 2.0;
            let curve = |stops| constant_curve(stops, 1.0 / 64.0);
            meta.channels = if records == 1 {
                [curve(-1.0); 3]
            } else {
                [curve(-1.0), curve(1.0), curve(2.0)]
            };
            let format = ultrahdr_core::Iso21496Format::JpegApp2BodyWithUrn;
            let wire = ultrahdr_core::serialize_iso21496_fmt(&meta, format).unwrap();
            let parsed = ultrahdr_core::parse_iso21496_fmt(&wire, format).unwrap();
            assert_eq!(parsed.iso_channel_count(), Some(records));
            let expected = if records == 1 {
                [0.1171875, 0.2421875, 0.3671875, 0.5]
            } else {
                [0.1171875, 1.015625, 3.046875, 0.5]
            };
            assert_renderers(&base, &gm, &parsed, 4.0, &[expected]);
        }
    }
}

#[test]
fn odd_center_uses_shepard_lattice_and_ignores_stride_padding() {
    let active = [[0.25, 0.5, 0.75, 0.5]; 9];
    let mut padded = [[9.0f32; 4]; 12];
    for row in padded.chunks_mut(4) {
        row[..3].copy_from_slice(&active[..3]);
    }
    let spec = descriptor_for(PixelFormat::RGBAF32, PRIMARIES, TransferFunction::Linear);
    let strided = PixelSlice::new(bytemuck::cast_slice(&padded), 3, 3, 64, &spec).unwrap();
    let gm = GainMap {
        width: 2,
        height: 2,
        channels: 1,
        data: vec![0, 85, 170, 255],
    };
    let mut meta = GainMapMetadata::default();
    meta.alternate_hdr_headroom = 3.0;
    meta.channels = [GainMapChannel {
        min: 0.0,
        max: 3.0,
        gamma: 1.0,
        base_offset: 0.0,
        alternate_offset: 0.0,
    }; 3];
    let fmt = HdrOutputFormat::LinearFloat;
    let tight = apply_gainmap(
        &linear_buffer(&active, 3, 3),
        &gm,
        &meta,
        8.0,
        fmt,
        Unstoppable,
    )
    .unwrap();
    let view = apply_gainmap_slice(strided, &gm, &meta, 8.0, fmt, Unstoppable).unwrap();
    assert_eq!(
        tight.as_view().as_strided_bytes(),
        view.as_view().as_strided_bytes()
    );
    // For the center (2/3, 2/3) in gain-map coordinates, the inverse-distance
    // weights are proportional to 1/sqrt(8), 1/sqrt(5), 1/sqrt(5), 1/sqrt(2).
    let weights = [8.0f64, 5.0, 5.0, 2.0].map(|squared| 1.0 / squared.sqrt());
    let gain = weights
        .into_iter()
        .zip([1.0, 2.0, 4.0, 8.0])
        .map(|(w, g)| w * g)
        .sum::<f64>()
        / weights.into_iter().sum::<f64>();
    let row = tight.as_view().row_as::<f32>(1).unwrap();
    for (i, input) in [0.25, 0.5, 0.75].into_iter().enumerate() {
        // Two ULPs bound the f32 kernel's rounded weights, LUT and additions
        // versus the independent f64 geometric reference.
        let expected = (input * gain) as f32;
        assert!(row[4 + i].to_bits().abs_diff(expected.to_bits()) <= 2);
    }
    assert_eq!(row[7], 0.5);
}

#[cfg(feature = "tonemap-bt2446a")]
#[test]
fn cll_distinguishes_native_max_rgb_and_bt2020_luminance() {
    use ultrahdr_core::color::audited::{
        DiffuseWhite, LightLevelMethod, measure_content_light_level, measure_peak,
    };
    let red = linear_buffer(&[[4.0, 0.0, 0.0, 1.0]], 1, 1);
    assert_eq!(
        measure_peak(&red, DiffuseWhite::BT2408, None).unwrap(),
        812.0
    );
    let frame = linear_buffer(&[[4.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], 2, 1);
    let cll =
        |method| measure_content_light_level(&frame, DiffuseWhite::BT2408, None, method).unwrap();
    let native = cll(LightLevelMethod::MaxRgb);
    let luma = cll(LightLevelMethod::LuminanceBt2020);
    assert_eq!(native.max_content_light_level, 812);
    assert_eq!(native.max_frame_average_light_level, 406);
    // The red sample's CIE Y uses the BT.709 red coefficient. The
    // reported CLL and FALL are integer upper bounds on max and mean.
    let red_y = 812.0_f64 * 0.2126;
    assert_eq!(luma.max_content_light_level, red_y.ceil() as u16);
    assert_eq!(
        luma.max_frame_average_light_level,
        (red_y / 2.0).ceil() as u16
    );
}

#[test]
fn exact_iso_headrooms_survive_equal_f64_projection() {
    // These fractions bracket the f32 log2 of the next display boost above
    // 2. Their distances to the target are 1/(2^23 * denominator).
    let base_den = 2_147_483_649u32;
    let alt_den = 2_155_872_255u32;
    let fields = [
        base_den + 256,
        base_den,
        alt_den + 257,
        alt_den,
        0,
        1,
        2,
        1,
        1,
        1,
        1,
        64,
        1,
        64,
    ];
    let mut wire = vec![0, 0, 0, 0, 0x40];
    for field in fields {
        wire.extend(field.to_be_bytes());
    }
    let format = ultrahdr_core::Iso21496Format::JxlJhgm;
    let metadata = ultrahdr_core::parse_iso21496_fmt(&wire, format).unwrap();
    assert_eq!(metadata.base_hdr_headroom, metadata.alternate_hdr_headroom);
    assert_eq!(
        ultrahdr_core::serialize_iso21496_fmt(&metadata, format).unwrap(),
        wire
    );

    let boost = 2.0f32.next_up();
    assert_eq!(boost.log2(), 1.0 + 1.0 / 8_388_608.0);
    let expected = (f64::from(alt_den) / (f64::from(base_den) + f64::from(alt_den))) as f32;
    assert_eq!(calculate_weight(boost, &metadata), expected);
}
