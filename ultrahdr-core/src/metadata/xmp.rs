//! XMP metadata serialization for Ultra HDR.
//!
//! Uses the Adobe HDR Gain Map namespace (hdrgm).

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::types::{Error, GainMapMetadata, Result};

/// XMP namespace for HDR gain map metadata.
pub const HDRGM_NAMESPACE: &str = "http://ns.adobe.com/hdr-gain-map/1.0/";

/// XMP namespace for container directory.
pub const CONTAINER_NAMESPACE: &str = "http://ns.google.com/photos/1.0/container/";

/// XMP namespace for container item.
pub const ITEM_NAMESPACE: &str = "http://ns.google.com/photos/1.0/container/item/";

/// Generate XMP metadata for Ultra HDR image.
///
/// This creates the XMP packet that goes in the primary JPEG's APP1 marker.
pub fn generate_xmp(metadata: &GainMapMetadata, gainmap_length: usize) -> String {
    let is_single_channel = metadata.is_single_channel();

    // Format values - use single value if all channels are the same
    let gain_map_min = format_value(&metadata.min_content_boost, is_single_channel, true);
    let gain_map_max = format_value(&metadata.max_content_boost, is_single_channel, true);
    let gamma = format_value(&metadata.gamma, is_single_channel, false);
    let offset_sdr = format_value(&metadata.offset_sdr, is_single_channel, false);
    let offset_hdr = format_value(&metadata.offset_hdr, is_single_channel, false);

    // Log2 of capacity values
    let hdr_capacity_min = metadata.hdr_capacity_min.log2();
    let hdr_capacity_max = metadata.hdr_capacity_max.log2();

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
      <Container:Directory>
        <rdf:Seq>
          <rdf:li rdf:parseType="Resource">
            <Container:Item
                Item:Semantic="Primary"
                Item:Mime="image/jpeg"/>
          </rdf:li>
          <rdf:li rdf:parseType="Resource">
            <Container:Item
                Item:Semantic="GainMap"
                Item:Mime="image/jpeg"
                Item:Length="{gainmap_length}"/>
          </rdf:li>
        </rdf:Seq>
      </Container:Directory>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"#
    )
}

/// Format a 3-element array as XMP value.
fn format_value(values: &[f32; 3], single_channel: bool, use_log2: bool) -> String {
    if single_channel {
        let v = if use_log2 {
            values[0].log2()
        } else {
            values[0]
        };
        format!("{:.6}", v)
    } else {
        let v: Vec<f32> = if use_log2 {
            values.iter().map(|x| x.log2()).collect()
        } else {
            values.to_vec()
        };
        format!("{:.6}, {:.6}, {:.6}", v[0], v[1], v[2])
    }
}

/// Parse XMP metadata from an Ultra HDR image.
pub fn parse_xmp(xmp_data: &str) -> Result<(GainMapMetadata, Option<usize>)> {
    // Check for hdrgm:Version
    if !xmp_data.contains("hdrgm:Version") && !xmp_data.contains("hdrgm:GainMapMax") {
        return Err(Error::NotUltraHdr);
    }

    let mut metadata = GainMapMetadata::new();
    let mut gainmap_length = None;

    // Parse hdrgm:GainMapMin
    if let Some(val) = extract_attribute(xmp_data, "hdrgm:GainMapMin") {
        let values = parse_xmp_values(&val);
        for (i, &v) in values.iter().enumerate() {
            metadata.min_content_boost[i] = 2.0f32.powf(v); // Convert from log2
        }
    }

    // Parse hdrgm:GainMapMax
    if let Some(val) = extract_attribute(xmp_data, "hdrgm:GainMapMax") {
        let values = parse_xmp_values(&val);
        for (i, &v) in values.iter().enumerate() {
            metadata.max_content_boost[i] = 2.0f32.powf(v); // Convert from log2
        }
    }

    // Parse hdrgm:Gamma
    if let Some(val) = extract_attribute(xmp_data, "hdrgm:Gamma") {
        let values = parse_xmp_values(&val);
        metadata.gamma = values;
    }

    // Parse hdrgm:OffsetSDR
    if let Some(val) = extract_attribute(xmp_data, "hdrgm:OffsetSDR") {
        metadata.offset_sdr = parse_xmp_values(&val);
    }

    // Parse hdrgm:OffsetHDR
    if let Some(val) = extract_attribute(xmp_data, "hdrgm:OffsetHDR") {
        metadata.offset_hdr = parse_xmp_values(&val);
    }

    // Parse hdrgm:HDRCapacityMin
    if let Some(val) = extract_attribute(xmp_data, "hdrgm:HDRCapacityMin") {
        if let Ok(v) = val.parse::<f32>() {
            metadata.hdr_capacity_min = 2.0f32.powf(v);
        }
    }

    // Parse hdrgm:HDRCapacityMax
    if let Some(val) = extract_attribute(xmp_data, "hdrgm:HDRCapacityMax") {
        if let Ok(v) = val.parse::<f32>() {
            metadata.hdr_capacity_max = 2.0f32.powf(v);
        }
    }

    // Parse Item:Length for gain map
    if let Some(val) = extract_attribute(xmp_data, "Item:Length") {
        if let Ok(len) = val.parse::<usize>() {
            gainmap_length = Some(len);
        }
    }

    Ok((metadata, gainmap_length))
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
fn parse_xmp_values(value: &str) -> [f32; 3] {
    let parsed: Vec<f32> = value
        .split(',')
        .filter_map(|s| s.trim().parse::<f32>().ok())
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
            min_content_boost: [1.0; 3],
            max_content_boost: [4.0; 3],
            gamma: [1.0; 3],
            offset_sdr: [0.015625; 3],
            offset_hdr: [0.015625; 3],
            hdr_capacity_min: 1.0,
            hdr_capacity_max: 4.0,
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
            min_content_boost: [1.0; 3],
            max_content_boost: [4.0; 3],
            gamma: [1.0; 3],
            offset_sdr: [0.015625; 3],
            offset_hdr: [0.015625; 3],
            hdr_capacity_min: 1.0,
            hdr_capacity_max: 4.0,
            use_base_color_space: true,
        };

        let xmp = generate_xmp(&original, 5000);
        let (parsed, length) = parse_xmp(&xmp).unwrap();

        assert_eq!(length, Some(5000));

        // Check values match (with some tolerance for log2 conversion)
        assert!((parsed.max_content_boost[0] - 4.0).abs() < 0.01);
        assert!((parsed.hdr_capacity_max - 4.0).abs() < 0.01);
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
            min_content_boost: [1.0, 2.0, 3.0],
            max_content_boost: [4.0, 5.0, 6.0],
            gamma: [1.0, 1.2, 1.5],
            offset_sdr: [0.015625; 3],
            offset_hdr: [0.015625; 3],
            hdr_capacity_min: 1.0,
            hdr_capacity_max: 6.0,
            use_base_color_space: true,
        };

        let xmp = generate_xmp(&metadata, 5000);
        let (parsed, _) = parse_xmp(&xmp).unwrap();

        // Verify channels differ for min_content_boost
        assert!(
            (parsed.min_content_boost[0] - parsed.min_content_boost[1]).abs() > 0.01,
            "channels 0 and 1 should differ"
        );
        assert!(
            (parsed.min_content_boost[1] - parsed.min_content_boost[2]).abs() > 0.01,
            "channels 1 and 2 should differ"
        );

        // Verify approximate values (log2 roundtrip)
        assert!((parsed.min_content_boost[0] - 1.0).abs() < 0.01);
        assert!((parsed.min_content_boost[1] - 2.0).abs() < 0.01);
        assert!((parsed.min_content_boost[2] - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_parse_xmp_values_empty() {
        assert_eq!(parse_xmp_values(""), [0.0; 3]);
    }

    #[test]
    fn test_parse_xmp_values_single() {
        assert_eq!(parse_xmp_values("1.5"), [1.5, 1.5, 1.5]);
    }

    #[test]
    fn test_parse_xmp_values_three() {
        assert_eq!(parse_xmp_values("1.0, 2.0, 3.0"), [1.0, 2.0, 3.0]);
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
            min_content_boost: [1.0; 3],
            max_content_boost: [4.0; 3],
            gamma: [1.0; 3],
            offset_sdr: [0.015625; 3],
            offset_hdr: [0.015625; 3],
            hdr_capacity_min: 1.0,
            hdr_capacity_max: 4.0,
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
