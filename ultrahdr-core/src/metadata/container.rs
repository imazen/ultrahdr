//! GContainer and MPF types shared across Ultra HDR, depth maps, and
//! other multi-image JPEG formats.
//!
//! These types model the two container mechanisms found in multi-image JPEGs:
//!
//! - **GContainer** (Google/Adobe XMP): `Container:Directory` with `Item:Semantic`
//!   values like `"Primary"`, `"GainMap"`, `"DepthMap"`, etc.
//!
//! - **MPF** (CIPA DC-007): Multi-Picture Format directory with typed image entries
//!   identified by `MpImageType` codes (primary, disparity, panorama, etc.).

use alloc::string::String;
use alloc::vec::Vec;

// ============================================================================
// GContainer types (XMP Container:Directory)
// ============================================================================

/// Semantic role of an item in a GContainer directory.
///
/// These correspond to `Item:Semantic` values in the XMP `Container:Directory`.
/// Used by Ultra HDR, Android Dynamic Depth Format, and Google Camera.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ItemSemantic {
    /// Primary image (`"Primary"`).
    Primary,
    /// HDR gain map (`"GainMap"`).
    GainMap,
    /// Depth map (`"DepthMap"`).
    DepthMap,
    /// Confidence/quality map for depth (`"ConfidenceMap"`).
    ConfidenceMap,
    /// Unrecognized or vendor-specific semantic.
    Other(String),
}

impl ItemSemantic {
    /// Parse from XMP `Item:Semantic` string value.
    pub fn from_xmp(s: &str) -> Self {
        match s {
            "Primary" => Self::Primary,
            "GainMap" => Self::GainMap,
            "DepthMap" => Self::DepthMap,
            "ConfidenceMap" => Self::ConfidenceMap,
            other => Self::Other(String::from(other)),
        }
    }

    /// XMP string representation.
    pub fn as_xmp_str(&self) -> &str {
        match self {
            Self::Primary => "Primary",
            Self::GainMap => "GainMap",
            Self::DepthMap => "DepthMap",
            Self::ConfidenceMap => "ConfidenceMap",
            Self::Other(s) => s.as_str(),
        }
    }
}

/// An item in a GContainer directory.
///
/// Parsed from `Container:Directory` → `rdf:Seq` → `rdf:li` entries in XMP.
/// Each item describes a secondary image appended after the primary JPEG's EOI.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContainerItem {
    /// Semantic role of this item.
    pub semantic: ItemSemantic,
    /// MIME type (e.g., `"image/jpeg"`, `"image/png"`).
    pub mime: String,
    /// Byte length of this item's data. `None` for the primary image.
    pub length: Option<usize>,
    /// Padding before this item (bytes). Used by some DDF implementations.
    pub padding: Option<usize>,
}

impl ContainerItem {
    /// Create a primary image item.
    pub fn primary(mime: impl Into<String>) -> Self {
        Self {
            semantic: ItemSemantic::Primary,
            mime: mime.into(),
            length: None,
            padding: None,
        }
    }

    /// Create a secondary item with a known length.
    pub fn secondary(semantic: ItemSemantic, mime: impl Into<String>, length: usize) -> Self {
        Self {
            semantic,
            mime: mime.into(),
            length: Some(length),
            padding: None,
        }
    }
}

/// Parse all `Container:Directory` items from XMP.
///
/// Returns items in order. The first item is typically `Primary`.
/// This parses the `rdf:Seq` inside `Container:Directory`, extracting
/// `Item:Semantic`, `Item:Mime`, `Item:Length`, and `Item:Padding`.
pub fn parse_container_items(xmp: &str) -> Vec<ContainerItem> {
    let mut items = Vec::new();

    // Find each <rdf:li rdf:parseType="Resource"> block inside Container:Directory
    let mut search_from = 0;
    while let Some(li_start) = xmp[search_from..].find("rdf:li") {
        let abs_start = search_from + li_start;
        // Find the end of this rdf:li block (either </rdf:li> or next rdf:li or </rdf:Seq>)
        let block_end = xmp[abs_start..]
            .find("</rdf:li>")
            .map(|p| abs_start + p)
            .or_else(|| {
                xmp[abs_start + 6..]
                    .find("rdf:li")
                    .map(|p| abs_start + 6 + p)
            })
            .unwrap_or(xmp.len());

        let block = &xmp[abs_start..block_end];

        // Extract Item:Semantic
        let semantic =
            extract_attr_from_block(block, "Item:Semantic").map(|s| ItemSemantic::from_xmp(&s));

        // Extract Item:Mime
        let mime = extract_attr_from_block(block, "Item:Mime");

        if let (Some(semantic), Some(mime)) = (semantic, mime) {
            let length =
                extract_attr_from_block(block, "Item:Length").and_then(|s| s.parse::<usize>().ok());
            let padding = extract_attr_from_block(block, "Item:Padding")
                .and_then(|s| s.parse::<usize>().ok());

            items.push(ContainerItem {
                semantic,
                mime,
                length,
                padding,
            });
        }

        search_from = block_end;
    }

    items
}

/// Generate `Container:Directory` XML fragment for a list of items.
///
/// The returned string is an `rdf:Seq` block suitable for embedding inside
/// an `rdf:Description` element that declares the Container and Item namespaces.
pub fn generate_container_directory(items: &[ContainerItem]) -> String {
    let mut xml = String::from("      <Container:Directory>\n        <rdf:Seq>\n");

    for item in items {
        xml.push_str("          <rdf:li rdf:parseType=\"Resource\">\n");
        xml.push_str("            <Container:Item\n");
        xml.push_str(&alloc::format!(
            "                Item:Semantic=\"{}\"\n",
            item.semantic.as_xmp_str()
        ));
        xml.push_str(&alloc::format!(
            "                Item:Mime=\"{}\"",
            item.mime
        ));
        if let Some(length) = item.length {
            xml.push_str(&alloc::format!(
                "\n                Item:Length=\"{}\"",
                length
            ));
        }
        if let Some(padding) = item.padding {
            xml.push_str(&alloc::format!(
                "\n                Item:Padding=\"{}\"",
                padding
            ));
        }
        xml.push_str("/>\n");
        xml.push_str("          </rdf:li>\n");
    }

    xml.push_str("        </rdf:Seq>\n      </Container:Directory>");
    xml
}

/// Extract an attribute value from a block of XMP text.
fn extract_attr_from_block(block: &str, attr_name: &str) -> Option<String> {
    let pattern = alloc::format!("{}=\"", attr_name);
    if let Some(start) = block.find(&pattern) {
        let value_start = start + pattern.len();
        if let Some(end) = block[value_start..].find('"') {
            return Some(String::from(&block[value_start..value_start + end]));
        }
    }
    None
}

// ============================================================================
// MPF types (CIPA DC-007)
// ============================================================================

/// Multi-Picture Format image type (CIPA DC-007 Individual Image Attribute).
///
/// The type code occupies bits 16-23 of the MP Entry attribute field.
/// Reconciled with zenjpeg's `MpfImageType` for cross-crate compatibility.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum MpImageType {
    /// Undefined type (0x000000). Used for gain maps in UltraHDR.
    #[default]
    Undefined,
    /// Baseline MP primary image (0x030000).
    BaselinePrimary,
    /// Large thumbnail — VGA resolution (0x010001).
    LargeThumbnailVga,
    /// Large thumbnail — Full HD resolution (0x010002).
    LargeThumbnailFullHd,
    /// Multi-frame panorama (0x020001).
    Panorama,
    /// Multi-frame disparity / depth map (0x020002).
    Disparity,
    /// Multi-frame multi-angle (0x020003).
    MultiAngle,
    /// Unrecognized type code.
    Other(u32),
}

impl MpImageType {
    /// Create from MPF type code (Individual Image Attribute field).
    pub fn from_type_code(code: u32) -> Self {
        match code {
            0x000000 => Self::Undefined,
            0x030000 => Self::BaselinePrimary,
            0x010001 => Self::LargeThumbnailVga,
            0x010002 => Self::LargeThumbnailFullHd,
            0x020001 => Self::Panorama,
            0x020002 => Self::Disparity,
            0x020003 => Self::MultiAngle,
            other => Self::Other(other),
        }
    }

    /// MPF type code for the attribute field.
    pub fn type_code(self) -> u32 {
        match self {
            Self::Undefined => 0x000000,
            Self::BaselinePrimary => 0x030000,
            Self::LargeThumbnailVga => 0x010001,
            Self::LargeThumbnailFullHd => 0x010002,
            Self::Panorama => 0x020001,
            Self::Disparity => 0x020002,
            Self::MultiAngle => 0x020003,
            Self::Other(code) => code,
        }
    }
}

/// A parsed MPF directory entry.
///
/// Represents one image in the MPF directory with its type, byte offset,
/// and size within the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MpfEntry {
    /// Image type from the attribute field.
    pub image_type: MpImageType,
    /// Absolute byte offset of this image in the file.
    /// For the primary image, this is always 0.
    pub offset: usize,
    /// Size of this image in bytes.
    pub size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_item_semantic_roundtrip() {
        for s in ["Primary", "GainMap", "DepthMap", "ConfidenceMap"] {
            let parsed = ItemSemantic::from_xmp(s);
            assert_eq!(parsed.as_xmp_str(), s);
        }
        let other = ItemSemantic::from_xmp("CustomThing");
        assert_eq!(other.as_xmp_str(), "CustomThing");
    }

    #[test]
    fn test_container_item_constructors() {
        let primary = ContainerItem::primary("image/jpeg");
        assert_eq!(primary.semantic, ItemSemantic::Primary);
        assert!(primary.length.is_none());

        let gainmap = ContainerItem::secondary(ItemSemantic::GainMap, "image/jpeg", 5000);
        assert_eq!(gainmap.length, Some(5000));
    }

    #[test]
    fn test_parse_container_items_ultrahdr() {
        let xmp = r#"
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
                        Item:Length="5000"/>
                </rdf:li>
            </rdf:Seq>
        </Container:Directory>"#;

        let items = parse_container_items(xmp);
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].semantic, ItemSemantic::Primary);
        assert_eq!(items[0].mime, "image/jpeg");
        assert!(items[0].length.is_none());
        assert_eq!(items[1].semantic, ItemSemantic::GainMap);
        assert_eq!(items[1].length, Some(5000));
    }

    #[test]
    fn test_parse_container_items_dynamic_depth() {
        let xmp = r#"
        <Container:Directory>
            <rdf:Seq>
                <rdf:li rdf:parseType="Resource">
                    <Container:Item Item:Semantic="Primary" Item:Mime="image/jpeg"/>
                </rdf:li>
                <rdf:li rdf:parseType="Resource">
                    <Container:Item Item:Semantic="DepthMap" Item:Mime="image/jpeg" Item:Length="8000"/>
                </rdf:li>
                <rdf:li rdf:parseType="Resource">
                    <Container:Item Item:Semantic="ConfidenceMap" Item:Mime="image/jpeg" Item:Length="3000"/>
                </rdf:li>
            </rdf:Seq>
        </Container:Directory>"#;

        let items = parse_container_items(xmp);
        assert_eq!(items.len(), 3);
        assert_eq!(items[0].semantic, ItemSemantic::Primary);
        assert_eq!(items[1].semantic, ItemSemantic::DepthMap);
        assert_eq!(items[1].length, Some(8000));
        assert_eq!(items[2].semantic, ItemSemantic::ConfidenceMap);
        assert_eq!(items[2].length, Some(3000));
    }

    #[test]
    fn test_generate_container_directory_roundtrip() {
        let items = vec![
            ContainerItem::primary("image/jpeg"),
            ContainerItem::secondary(ItemSemantic::GainMap, "image/jpeg", 5000),
        ];
        let xml = generate_container_directory(&items);
        let parsed = parse_container_items(&xml);
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].semantic, ItemSemantic::Primary);
        assert_eq!(parsed[1].semantic, ItemSemantic::GainMap);
        assert_eq!(parsed[1].length, Some(5000));
    }

    #[test]
    fn test_generate_container_directory_multi_item() {
        let items = vec![
            ContainerItem::primary("image/jpeg"),
            ContainerItem::secondary(ItemSemantic::GainMap, "image/jpeg", 5000),
            ContainerItem::secondary(ItemSemantic::DepthMap, "image/png", 12000),
        ];
        let xml = generate_container_directory(&items);
        assert!(xml.contains("GainMap"));
        assert!(xml.contains("DepthMap"));
        assert!(xml.contains("image/png"));
        assert!(xml.contains("12000"));
    }

    #[test]
    fn test_mp_image_type_roundtrip() {
        let types = [
            (0x000000, MpImageType::Undefined),
            (0x030000, MpImageType::BaselinePrimary),
            (0x010001, MpImageType::LargeThumbnailVga),
            (0x020002, MpImageType::Disparity),
            (0x020003, MpImageType::MultiAngle),
            (0xDEAD00, MpImageType::Other(0xDEAD00)),
        ];
        for (code, expected) in types {
            let parsed = MpImageType::from_type_code(code);
            assert_eq!(parsed, expected);
            assert_eq!(parsed.type_code(), code);
        }
    }

    #[test]
    fn test_mpf_entry() {
        let entry = MpfEntry {
            image_type: MpImageType::Disparity,
            offset: 50000,
            size: 10000,
        };
        assert_eq!(entry.image_type, MpImageType::Disparity);
        assert_eq!(entry.offset, 50000);
        assert_eq!(entry.size, 10000);
    }
}
