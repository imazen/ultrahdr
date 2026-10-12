//! Minimal Apple binary property list (`bplist00`) reader.
//!
//! Several Apple MakerNote tags (e.g. `RunTime`, auto-exposure state) store an
//! embedded `bplist00` blob rather than a plain TIFF value. This reader decodes
//! the common object types well enough to inspect those structures. It is a
//! reader only — there is no writer — and it is deliberately small.
//!
//! Format reference: CoreFoundation `CFBinaryPList` (`bplist00`):
//! 8-byte `"bplist00"` header, packed objects, an offset table, and a 32-byte
//! trailer giving the integer widths, object count, top object index, and the
//! offset-table location.
//!
//! `no_std` + `alloc`, `#![forbid(unsafe_code)]`. Recursion is depth-bounded so
//! a maliciously cyclic ref graph cannot blow the stack.

use alloc::string::String;
use alloc::vec::Vec;

/// A decoded property-list value.
#[derive(Clone, Debug, PartialEq)]
pub enum PlistValue {
    /// Boolean (`0x08` false / `0x09` true).
    Bool(bool),
    /// Signed integer (1/2/4/8-byte; wider ints are truncated to the low 64 bits).
    Integer(i64),
    /// IEEE-754 real (`f32` widened to `f64`, or native `f64`).
    Real(f64),
    /// Date: seconds since the Apple epoch (2001-01-01T00:00:00Z).
    Date(f64),
    /// Raw data blob.
    Data(Vec<u8>),
    /// UTF-8 (from ASCII) or UTF-16BE string.
    String(String),
    /// CoreFoundation keyed-archiver UID.
    Uid(u64),
    /// Ordered array.
    Array(Vec<PlistValue>),
    /// Ordered key/value dictionary (insertion order preserved).
    Dict(Vec<(String, PlistValue)>),
}

/// Parse a `bplist00` blob within the shared parser's supported subset and
/// resource bounds. The published `PlistValue` type remains owned by this crate.
pub fn parse_bplist(data: &[u8]) -> Option<PlistValue> {
    from_shared(zenmeta::bplist::parse_bplist(data)?)
}

fn from_shared(value: zenmeta::bplist::PlistValue) -> Option<PlistValue> {
    use zenmeta::bplist::PlistValue as Shared;
    Some(match value {
        Shared::Bool(v) => PlistValue::Bool(v),
        Shared::Integer(v) => PlistValue::Integer(v),
        Shared::Real(v) => PlistValue::Real(v),
        Shared::Date(v) => PlistValue::Date(v),
        Shared::Data(v) => PlistValue::Data(v),
        Shared::String(v) => PlistValue::String(v),
        Shared::Uid(v) => PlistValue::Uid(v),
        Shared::Array(values) => {
            let mut out = Vec::new();
            out.try_reserve_exact(values.len()).ok()?;
            for value in values {
                out.push(from_shared(value)?);
            }
            PlistValue::Array(out)
        }
        Shared::Dict(values) => {
            let mut out = Vec::new();
            out.try_reserve_exact(values.len()).ok()?;
            for (key, value) in values {
                out.push((key, from_shared(value)?));
            }
            PlistValue::Dict(out)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `{"a": true, "b": 5}` encoded as a real bplist00 (built by Apple's
    /// `plutil`). Verifies header, trailer, dict, bool, and integer decode.
    #[test]
    fn parses_simple_dict() {
        // Hand-assembled bplist00:
        //  obj0 dict(2): keys [obj1,obj2] vals [obj3,obj4]
        //  obj1 "a" obj2 "b" obj3 true obj4 int(5)
        let mut d = Vec::new();
        d.extend_from_slice(b"bplist00");
        let o0 = d.len();
        d.push(0xD2); // dict, count 2
        d.extend_from_slice(&[1, 2]); // key refs
        d.extend_from_slice(&[3, 4]); // val refs
        let o1 = d.len();
        d.extend_from_slice(&[0x51, b'a']); // ASCII "a"
        let o2 = d.len();
        d.extend_from_slice(&[0x51, b'b']); // ASCII "b"
        let o3 = d.len();
        d.push(0x09); // true
        let o4 = d.len();
        d.extend_from_slice(&[0x10, 5]); // int 5 (1 byte)
        let table = d.len();
        for off in [o0, o1, o2, o3, o4] {
            d.push(off as u8); // offset_size = 1
        }
        // trailer (32 bytes)
        d.extend_from_slice(&[0; 6]); // unused + sortVersion
        d.push(1); // offset int size
        d.push(1); // object ref size
        d.extend_from_slice(&(5u64).to_be_bytes()); // num objects
        d.extend_from_slice(&(0u64).to_be_bytes()); // top object
        d.extend_from_slice(&(table as u64).to_be_bytes()); // offset table off

        let v = parse_bplist(&d).unwrap();
        let PlistValue::Dict(entries) = v else {
            panic!("expected dict, got {v:?}");
        };
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].0, "a");
        assert_eq!(entries[0].1, PlistValue::Bool(true));
        assert_eq!(entries[1].0, "b");
        assert_eq!(entries[1].1, PlistValue::Integer(5));
    }

    #[test]
    fn rejects_non_bplist() {
        assert_eq!(parse_bplist(b"not a plist at all..............."), None);
        assert_eq!(parse_bplist(b"bplist00"), None); // no trailer
        assert_eq!(parse_bplist(&[0u8; 4]), None);
    }
}
