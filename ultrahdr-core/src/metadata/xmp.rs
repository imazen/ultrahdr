//! XMP metadata serialization for Ultra HDR.
//!
//! Uses the Adobe HDR Gain Map namespace (hdrgm).

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::types::{Error, GainMapMetadata, Result};

use super::container::{
    ContainerItem, ItemSemantic, generate_container_directory, parse_container_items,
};

/// XMP namespace for HDR gain map metadata.
pub const HDRGM_NAMESPACE: &str = "http://ns.adobe.com/hdr-gain-map/1.0/";

/// XMP namespace for container directory.
pub const CONTAINER_NAMESPACE: &str = "http://ns.google.com/photos/1.0/container/";

/// XMP namespace for container item.
pub const ITEM_NAMESPACE: &str = "http://ns.google.com/photos/1.0/container/item/";

/// Generate XMP metadata for Ultra HDR image (convenience for 2-item Primary+GainMap).
///
/// This creates the XMP packet that goes in the primary JPEG's APP1 marker.
/// For multi-item containers (e.g., gain map + depth map), use
/// [`generate_xmp_with_items`] instead.
pub fn generate_xmp(metadata: &GainMapMetadata, gainmap_length: usize) -> String {
    let items = alloc::vec![
        ContainerItem::primary("image/jpeg"),
        ContainerItem::secondary(ItemSemantic::GainMap, "image/jpeg", gainmap_length),
    ];
    generate_xmp_with_items(metadata, &items)
}

/// Generate XMP metadata with a custom container directory.
///
/// Use this when the JPEG contains more than the standard Primary+GainMap
/// pair — for example, a depth map or confidence map alongside the gain map.
pub fn generate_xmp_with_items(metadata: &GainMapMetadata, items: &[ContainerItem]) -> String {
    let is_single_channel = metadata.is_single_channel();

    // GainMapMin/Max are already in log2 domain — write directly
    let gain_map_min = format_f64_value(&metadata.gain_map_min, is_single_channel);
    let gain_map_max = format_f64_value(&metadata.gain_map_max, is_single_channel);
    let gamma = format_f64_value(&metadata.gamma, is_single_channel);
    let offset_sdr = format_f64_value(&metadata.base_offset, is_single_channel);
    let offset_hdr = format_f64_value(&metadata.alternate_offset, is_single_channel);

    // Headroom values are already in log2 domain
    let hdr_capacity_min = metadata.base_hdr_headroom;
    let hdr_capacity_max = metadata.alternate_hdr_headroom;

    let container_dir = generate_container_directory(items);

    format!(
        r#"<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Adobe XMP Core">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
        xmlns:hdrgm="{HDRGM_NAMESPACE}"
        xmlns:Container="{CONTAINER_NAMESPACE}"
        xmlns:Item="{ITEM_NAMESPACE}"
        hdrgm:Version="1.0"
        hdrgm:GainMapMin="{gain_map_min}"
        hdrgm:GainMapMax="{gain_map_max}"
        hdrgm:Gamma="{gamma}"
        hdrgm:OffsetSDR="{offset_sdr}"
        hdrgm:OffsetHDR="{offset_hdr}"
        hdrgm:HDRCapacityMin="{hdr_capacity_min:.6}"
        hdrgm:HDRCapacityMax="{hdr_capacity_max:.6}"
        hdrgm:BaseRenditionIsHDR="False">
{container_dir}
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"#
    )
}

/// Format a 3-element f64 array as XMP value.
fn format_f64_value(values: &[f64; 3], single_channel: bool) -> String {
    if single_channel {
        format!("{:.6}", values[0])
    } else {
        format!("{:.6}, {:.6}, {:.6}", values[0], values[1], values[2])
    }
}

/// Parse XMP metadata from an Ultra HDR image.
///
/// Returns `(metadata, gainmap_length)` for backward compatibility.
/// For access to all container items (depth maps, etc.), use [`parse_xmp_full`].
pub fn parse_xmp(xmp_data: &str) -> Result<(GainMapMetadata, Option<usize>)> {
    // Check for hdrgm:Version
    if !xmp_data.contains("hdrgm:Version") && !xmp_data.contains("hdrgm:GainMapMax") {
        return Err(Error::NotUltraHdr);
    }

    let mut metadata = GainMapMetadata::new();
    let mut gainmap_length = None;

    // Parse hdrgm:GainMapMin — already log2 on wire, matches our domain
    if let Some(val) = extract_attribute(xmp_data, "hdrgm:GainMapMin") {
        metadata.gain_map_min = parse_xmp_values_f64(&val);
    }

    // Parse hdrgm:GainMapMax — already log2 on wire
    if let Some(val) = extract_attribute(xmp_data, "hdrgm:GainMapMax") {
        metadata.gain_map_max = parse_xmp_values_f64(&val);
    }

    // Parse hdrgm:Gamma — linear domain
    if let Some(val) = extract_attribute(xmp_data, "hdrgm:Gamma") {
        metadata.gamma = parse_xmp_values_f64(&val);
    }

    // Parse hdrgm:OffsetSDR — linear domain
    if let Some(val) = extract_attribute(xmp_data, "hdrgm:OffsetSDR") {
        metadata.base_offset = parse_xmp_values_f64(&val);
    }

    // Parse hdrgm:OffsetHDR — linear domain
    if let Some(val) = extract_attribute(xmp_data, "hdrgm:OffsetHDR") {
        metadata.alternate_offset = parse_xmp_values_f64(&val);
    }

    // Parse hdrgm:HDRCapacityMin — already log2 on wire
    if let Some(val) = extract_attribute(xmp_data, "hdrgm:HDRCapacityMin")
        && let Ok(v) = val.parse::<f64>()
    {
        metadata.base_hdr_headroom = v;
    }

    // Parse hdrgm:HDRCapacityMax — already log2 on wire
    if let Some(val) = extract_attribute(xmp_data, "hdrgm:HDRCapacityMax")
        && let Ok(v) = val.parse::<f64>()
    {
        metadata.alternate_hdr_headroom = v;
    }

    // Parse Item:Length for gain map
    if let Some(val) = extract_attribute(xmp_data, "Item:Length")
        && let Ok(len) = val.parse::<usize>()
    {
        gainmap_length = Some(len);
    }

    Ok((metadata, gainmap_length))
}

/// Parse XMP metadata and all container directory items.
///
/// Returns gain map metadata (if hdrgm namespace is present) plus all items
/// from the `Container:Directory`. Use this when you need to find depth maps,
/// confidence maps, or other secondary images alongside the gain map.
pub fn parse_xmp_full(xmp_data: &str) -> (Option<GainMapMetadata>, Vec<ContainerItem>) {
    let metadata = if xmp_data.contains("hdrgm:Version") || xmp_data.contains("hdrgm:GainMapMax") {
        parse_xmp(xmp_data).ok().map(|(m, _)| m)
    } else {
        None
    };

    let items = parse_container_items(xmp_data);
    (metadata, items)
}

/// Extract an attribute value from XMP using simple string matching.
fn extract_attribute(xmp: &str, attr_name: &str) -> Option<String> {
    // Try attribute format: attr="value"
    let pattern = format!("{}=\"", attr_name);
    if let Some(start) = xmp.find(&pattern) {
        let value_start = start + pattern.len();
        if let Some(end) = xmp[value_start..].find('"') {
            return Some(xmp[value_start..value_start + end].to_string());
        }
    }

    // Try element format: <attr>value</attr>
    let open_tag = format!("<{}>", attr_name);
    let close_tag = format!("</{}>", attr_name);
    if let Some(start) = xmp.find(&open_tag) {
        let value_start = start + open_tag.len();
        if let Some(end) = xmp[value_start..].find(&close_tag) {
            return Some(xmp[value_start..value_start + end].trim().to_string());
        }
    }

    None
}

/// Parse comma-separated or single values from XMP.
/// Returns exactly 3 values: if input has 1 value, it's replicated to all channels.
fn parse_xmp_values_f64(value: &str) -> [f64; 3] {
    let parsed: Vec<f64> = value
        .split(',')
        .filter_map(|s| s.trim().parse::<f64>().ok())
        .collect();

    match parsed.len() {
        0 => [0.0; 3],
        1 => [parsed[0]; 3], // Single value: replicate to all channels
        2 => [parsed[0], parsed[1], 0.0],
        _ => [parsed[0], parsed[1], parsed[2]], // 3+ values
    }
}

/// Create APP1 marker with XMP data.
pub fn create_xmp_app1_marker(xmp: &str) -> Vec<u8> {
    let xmp_bytes = xmp.as_bytes();
    let namespace = b"http://ns.adobe.com/xap/1.0/\0";

    // APP1 marker: FF E1 + length (2 bytes) + namespace + XMP data
    let total_length = 2 + namespace.len() + xmp_bytes.len();

    let mut marker = Vec::with_capacity(2 + total_length);
    marker.push(0xFF);
    marker.push(0xE1);
    marker.push(((total_length >> 8) & 0xFF) as u8);
    marker.push((total_length & 0xFF) as u8);
    marker.extend_from_slice(namespace);
    marker.extend_from_slice(xmp_bytes);

    marker
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_xmp() {
        let metadata = GainMapMetadata {
            gain_map_min: [0.0; 3],
            gain_map_max: [2.0; 3],
            gamma: [1.0; 3],
            base_offset: [0.015625; 3],
            alternate_offset: [0.015625; 3],
            base_hdr_headroom: 0.0,
            alternate_hdr_headroom: 2.0,
            use_base_color_space: true,
        };

        let xmp = generate_xmp(&metadata, 10000);

        assert!(xmp.contains("hdrgm:Version=\"1.0\""));
        assert!(xmp.contains("hdrgm:GainMapMax"));
        assert!(xmp.contains("Item:Length=\"10000\""));
        assert!(xmp.contains("Item:Semantic=\"GainMap\""));
    }

    #[test]
    fn test_parse_xmp_roundtrip() {
        let original = GainMapMetadata {
            gain_map_min: [0.0; 3],
            gain_map_max: [2.0; 3],
            gamma: [1.0; 3],
            base_offset: [0.015625; 3],
            alternate_offset: [0.015625; 3],
            base_hdr_headroom: 0.0,
            alternate_hdr_headroom: 2.0,
            use_base_color_space: true,
        };

        let xmp = generate_xmp(&original, 5000);
        let (parsed, length) = parse_xmp(&xmp).unwrap();

        assert_eq!(length, Some(5000));

        // Check values match (log2 domain roundtrip)
        assert!((parsed.gain_map_max[0] - 2.0).abs() < 0.01);
        assert!((parsed.alternate_hdr_headroom - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_extract_attribute() {
        let xmp = r#"<rdf:Description hdrgm:Version="1.0" hdrgm:GainMapMax="2.0"/>"#;

        assert_eq!(extract_attribute(xmp, "hdrgm:Version"), Some("1.0".into()));
        assert_eq!(
            extract_attribute(xmp, "hdrgm:GainMapMax"),
            Some("2.0".into())
        );
        assert_eq!(extract_attribute(xmp, "hdrgm:Missing"), None);
    }

    #[test]
    fn test_parse_xmp_not_ultrahdr() {
        // XMP without hdrgm namespace should return NotUltraHdr
        let xmp = r#"<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
        xmlns:dc="http://purl.org/dc/elements/1.1/"
        dc:creator="SomeCamera"/>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"#;

        let result = parse_xmp(xmp);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::NotUltraHdr));
    }

    #[test]
    fn test_parse_xmp_multi_channel() {
        // Generate XMP with different per-channel values
        let metadata = GainMapMetadata {
            gain_map_min: [1.0, 2.0, 3.0],
            gain_map_max: [4.0, 5.0, 6.0],
            gamma: [1.0, 1.2, 1.5],
            base_offset: [0.015625; 3],
            alternate_offset: [0.015625; 3],
            base_hdr_headroom: 0.0,
            alternate_hdr_headroom: 2.585,
            use_base_color_space: true,
        };

        let xmp = generate_xmp(&metadata, 5000);
        let (parsed, _) = parse_xmp(&xmp).unwrap();

        // Verify channels differ for min_content_boost
        assert!(
            (parsed.gain_map_min[0] - parsed.gain_map_min[1]).abs() > 0.01,
            "channels 0 and 1 should differ"
        );
        assert!(
            (parsed.gain_map_min[1] - parsed.gain_map_min[2]).abs() > 0.01,
            "channels 1 and 2 should differ"
        );

        // Verify approximate values (log2 roundtrip)
        assert!((parsed.gain_map_min[0] - 1.0).abs() < 0.01);
        assert!((parsed.gain_map_min[1] - 2.0).abs() < 0.01);
        assert!((parsed.gain_map_min[2] - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_parse_xmp_values_empty() {
        assert_eq!(parse_xmp_values_f64(""), [0.0; 3]);
    }

    #[test]
    fn test_parse_xmp_values_single() {
        assert_eq!(parse_xmp_values_f64("1.5"), [1.5, 1.5, 1.5]);
    }

    #[test]
    fn test_parse_xmp_values_three() {
        assert_eq!(parse_xmp_values_f64("1.0, 2.0, 3.0"), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_create_xmp_app1_marker() {
        let xmp = "<x:xmpmeta>test</x:xmpmeta>";
        let marker = create_xmp_app1_marker(xmp);

        // Starts with FF E1
        assert_eq!(marker[0], 0xFF);
        assert_eq!(marker[1], 0xE1);

        // Length field (bytes 2-3) is big-endian u16
        let length = u16::from_be_bytes([marker[2], marker[3]]) as usize;
        let namespace = b"http://ns.adobe.com/xap/1.0/\0";
        let expected_length = 2 + namespace.len() + xmp.len();
        assert_eq!(length, expected_length);

        // Contains XMP namespace after the length field
        let namespace_start = 4;
        let namespace_end = namespace_start + namespace.len();
        assert_eq!(&marker[namespace_start..namespace_end], namespace);

        // Total marker size = 2 (marker) + 2 (length) + namespace + xmp
        assert_eq!(marker.len(), 4 + namespace.len() + xmp.len());
    }

    #[test]
    fn test_extract_attribute_element_format() {
        // Test element format: <hdrgm:Gamma>1.0</hdrgm:Gamma>
        let xmp = r#"<rdf:Description>
  <hdrgm:Gamma>1.0</hdrgm:Gamma>
  <hdrgm:OffsetSDR>0.015625</hdrgm:OffsetSDR>
</rdf:Description>"#;

        assert_eq!(extract_attribute(xmp, "hdrgm:Gamma"), Some("1.0".into()));
        assert_eq!(
            extract_attribute(xmp, "hdrgm:OffsetSDR"),
            Some("0.015625".into())
        );
        assert_eq!(extract_attribute(xmp, "hdrgm:Missing"), None);
    }

    #[test]
    fn test_generate_xmp_contains_required_fields() {
        let metadata = GainMapMetadata {
            gain_map_min: [0.0; 3],
            gain_map_max: [2.0; 3],
            gamma: [1.0; 3],
            base_offset: [0.015625; 3],
            alternate_offset: [0.015625; 3],
            base_hdr_headroom: 0.0,
            alternate_hdr_headroom: 2.0,
            use_base_color_space: true,
        };

        let xmp = generate_xmp(&metadata, 8000);

        // All required hdrgm fields must be present
        assert!(xmp.contains("hdrgm:Version="), "missing Version");
        assert!(xmp.contains("hdrgm:GainMapMin="), "missing GainMapMin");
        assert!(xmp.contains("hdrgm:GainMapMax="), "missing GainMapMax");
        assert!(xmp.contains("hdrgm:Gamma="), "missing Gamma");
        assert!(xmp.contains("hdrgm:OffsetSDR="), "missing OffsetSDR");
        assert!(xmp.contains("hdrgm:OffsetHDR="), "missing OffsetHDR");
        assert!(
            xmp.contains("hdrgm:HDRCapacityMin="),
            "missing HDRCapacityMin"
        );
        assert!(
            xmp.contains("hdrgm:HDRCapacityMax="),
            "missing HDRCapacityMax"
        );
        assert!(
            xmp.contains("hdrgm:BaseRenditionIsHDR="),
            "missing BaseRenditionIsHDR"
        );

        // Namespace declarations
        assert!(xmp.contains(HDRGM_NAMESPACE), "missing hdrgm namespace");
        assert!(
            xmp.contains(CONTAINER_NAMESPACE),
            "missing container namespace"
        );
        assert!(xmp.contains(ITEM_NAMESPACE), "missing item namespace");

        // Container directory structure
        assert!(
            xmp.contains("Item:Semantic=\"Primary\""),
            "missing primary item"
        );
        assert!(
            xmp.contains("Item:Semantic=\"GainMap\""),
            "missing gainmap item"
        );
        assert!(
            xmp.contains("Item:Length=\"8000\""),
            "missing/wrong item length"
        );
    }
}
