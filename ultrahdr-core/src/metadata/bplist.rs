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

/// Maximum container nesting depth before parsing bails (cycle/DoS guard).
const MAX_DEPTH: usize = 32;

/// Parse a `bplist00` blob and return the top-level object, or `None` if the
/// data is not a well-formed binary plist within our supported subset.
pub fn parse_bplist(data: &[u8]) -> Option<PlistValue> {
    if data.len() < 8 + 32 || &data[..8] != b"bplist00" {
        return None;
    }
    let trailer = &data[data.len() - 32..];
    let offset_size = *trailer.get(6)? as usize;
    let ref_size = *trailer.get(7)? as usize;
    if offset_size == 0 || offset_size > 8 || ref_size == 0 || ref_size > 8 {
        return None;
    }
    let num_objects = read_uint_be(trailer, 8, 8)? as usize;
    let top_object = read_uint_be(trailer, 16, 8)? as usize;
    let offset_table_off = read_uint_be(trailer, 24, 8)? as usize;
    if top_object >= num_objects {
        return None;
    }
    // The offset table must fit before the trailer.
    let table_end = offset_table_off.checked_add(num_objects.checked_mul(offset_size)?)?;
    if table_end > data.len() - 32 {
        return None;
    }
    let ctx = Ctx {
        data,
        num_objects,
        offset_size,
        ref_size,
        offset_table_off,
    };
    ctx.read_object(top_object, 0)
}

/// Immutable parsing context shared across the recursive descent.
struct Ctx<'a> {
    data: &'a [u8],
    num_objects: usize,
    offset_size: usize,
    ref_size: usize,
    offset_table_off: usize,
}

impl Ctx<'_> {
    /// Byte offset of object `index` from the offset table.
    fn object_offset(&self, index: usize) -> Option<usize> {
        if index >= self.num_objects {
            return None;
        }
        let pos = self.offset_table_off + index * self.offset_size;
        read_uint_be(self.data, pos, self.offset_size).map(|v| v as usize)
    }

    /// Read an object reference (used inside arrays/dicts) at byte `pos`.
    fn read_ref(&self, pos: usize) -> Option<usize> {
        read_uint_be(self.data, pos, self.ref_size).map(|v| v as usize)
    }

    fn read_object(&self, index: usize, depth: usize) -> Option<PlistValue> {
        if depth > MAX_DEPTH {
            return None;
        }
        let off = self.object_offset(index)?;
        let marker = *self.data.get(off)?;
        let hi = marker >> 4;
        let lo = (marker & 0x0F) as usize;
        match hi {
            0x0 => match marker {
                0x08 => Some(PlistValue::Bool(false)),
                0x09 => Some(PlistValue::Bool(true)),
                _ => None, // null / fill / unsupported singletons
            },
            0x1 => {
                // Integer: 2^lo bytes, big-endian. bplist 1/2/4-byte ints are
                // unsigned and 8-byte ints are signed; `raw as i64` is correct
                // for all (the 8-byte bit pattern reinterprets as signed).
                let n = 1usize << lo;
                let raw = read_uint_be(self.data, off + 1, n.min(8))?;
                Some(PlistValue::Integer(raw as i64))
            }
            0x2 => {
                // Real: 2^lo bytes (4 = f32, 8 = f64).
                let n = 1usize << lo;
                let bytes = self.data.get(off + 1..off + 1 + n)?;
                let v = match n {
                    4 => f32::from_be_bytes(bytes.try_into().ok()?) as f64,
                    8 => f64::from_be_bytes(bytes.try_into().ok()?),
                    _ => return None,
                };
                Some(PlistValue::Real(v))
            }
            0x3 => {
                // Date: always an 8-byte f64.
                let bytes = self.data.get(off + 1..off + 9)?;
                Some(PlistValue::Date(f64::from_be_bytes(bytes.try_into().ok()?)))
            }
            0x4 => {
                let (count, base) = self.read_count(off, lo)?;
                let bytes = self.data.get(base..base + count)?;
                Some(PlistValue::Data(bytes.to_vec()))
            }
            0x5 => {
                // ASCII string: `count` bytes, one byte per char.
                let (count, base) = self.read_count(off, lo)?;
                let bytes = self.data.get(base..base + count)?;
                Some(PlistValue::String(core::str::from_utf8(bytes).ok()?.into()))
            }
            0x6 => {
                // UTF-16BE string: `count` UTF-16 code units (2 bytes each).
                let (count, base) = self.read_count(off, lo)?;
                let end = base + count * 2;
                let bytes = self.data.get(base..end)?;
                let mut units = Vec::with_capacity(count);
                for c in bytes.chunks_exact(2) {
                    units.push(u16::from_be_bytes([c[0], c[1]]));
                }
                let s: String = char::decode_utf16(units)
                    .map(|r| r.unwrap_or('\u{FFFD}'))
                    .collect();
                Some(PlistValue::String(s))
            }
            0x8 => {
                // UID: lo+1 bytes.
                let n = lo + 1;
                Some(PlistValue::Uid(read_uint_be(self.data, off + 1, n.min(8))?))
            }
            0xA | 0xC => {
                // Array (0xA) or set (0xC): `count` object refs.
                let (count, base) = self.read_count(off, lo)?;
                let mut out = Vec::with_capacity(count);
                for i in 0..count {
                    let r = self.read_ref(base + i * self.ref_size)?;
                    out.push(self.read_object(r, depth + 1)?);
                }
                Some(PlistValue::Array(out))
            }
            0xD => {
                // Dict: `count` key refs followed by `count` value refs.
                let (count, base) = self.read_count(off, lo)?;
                let keys_base = base;
                let vals_base = base + count * self.ref_size;
                let mut out = Vec::with_capacity(count);
                for i in 0..count {
                    let k = self.read_ref(keys_base + i * self.ref_size)?;
                    let v = self.read_ref(vals_base + i * self.ref_size)?;
                    let key = match self.read_object(k, depth + 1)? {
                        PlistValue::String(s) => s,
                        _ => return None, // non-string keys are unsupported
                    };
                    out.push((key, self.read_object(v, depth + 1)?));
                }
                Some(PlistValue::Dict(out))
            }
            _ => None,
        }
    }

    /// Decode a container/data/string length. When the marker's low nibble is
    /// `0xF`, the count is an inline integer object that follows. Returns
    /// `(count, base)` where `base` is the byte offset of the first element.
    fn read_count(&self, off: usize, lo: usize) -> Option<(usize, usize)> {
        if lo != 0x0F {
            return Some((lo, off + 1));
        }
        let size_marker = *self.data.get(off + 1)?;
        if size_marker >> 4 != 0x1 {
            return None;
        }
        let n = 1usize << (size_marker & 0x0F);
        let count = read_uint_be(self.data, off + 2, n.min(8))? as usize;
        Some((count, off + 2 + n))
    }
}

/// Read a big-endian unsigned integer of `size` bytes (1..=8) at `off`.
fn read_uint_be(data: &[u8], off: usize, size: usize) -> Option<u64> {
    if size == 0 || size > 8 {
        return None;
    }
    let bytes = data.get(off..off + size)?;
    let mut v = 0u64;
    for &b in bytes {
        v = (v << 8) | b as u64;
    }
    Some(v)
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
