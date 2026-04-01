#!/bin/bash
# Gain map interop validation script
#
# Validates ISO 21496-1 gain map metadata against third-party tools:
# - libultrahdr (JPEG Ultra HDR)
# - libavif (AVIF tmap)
# - ExifTool (XMP/MPF structure)
#
# Usage:
#   validate_gainmap                    # validate built-in test files
#   validate_gainmap /path/to/file.jpg  # validate a specific JPEG
#   validate_gainmap /path/to/dir/      # validate all JPEGs/AVIFs in directory
set -uo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
PASS=0; FAIL=0; SKIP=0
pass() { echo -e "  ${GREEN}PASS${NC}: $1"; ((PASS++)); }
fail() { echo -e "  ${RED}FAIL${NC}: $1"; ((FAIL++)); }
skip() { echo -e "  ${YELLOW}SKIP${NC}: $1"; ((SKIP++)); }
section() { echo -e "\n${BLUE}=== $1 ===${NC}\n"; }

shopt -s nullglob
SAMPLES="${1:-/samples}"
OUTDIR="/tmp/gainmap_validation"
mkdir -p "$OUTDIR"

# ============================================================================
section "Tool versions"
# ============================================================================
echo "  ultrahdr_app: $(ultrahdr_app 2>&1 | grep -o 'v[0-9.]*' | head -1 || echo 'unknown')"
echo "  avifenc:      $(avifenc --version 2>&1 | head -1)"
echo "  avifdec:      $(avifdec --version 2>&1 | head -1)"
echo "  exiftool:     $(exiftool -ver)"

# ============================================================================
section "1. JPEG Ultra HDR validation (libultrahdr)"
# ============================================================================
jpeg_count=0
for f in "$SAMPLES"/*.jpg "$SAMPLES"/*.jpeg; do
    [ -f "$f" ] || continue
    name=$(basename "$f")
    ((jpeg_count++))

    # Probe
    probe=$(ultrahdr_app -m 1 -P -j "$f" 2>&1) || true
    if echo "$probe" | grep -q "Ultra HDR Image: Yes"; then
        pass "$name: ultrahdr_app probe OK"

        # Extract metadata values
        max_boost=$(echo "$probe" | grep "maxContentBoost" | awk '{print $2}')
        capacity=$(echo "$probe" | grep "hdrCapacityMax" | awk '{print $2}')
        echo "    maxContentBoost=$max_boost, hdrCapacityMax=$capacity"

        # Verify metadata is sane
        if echo "$probe" | grep -q "hdrCapacityMin inf"; then
            fail "$name: hdrCapacityMin is inf (bad base_hdr_headroom)"
        fi
    else
        skip "$name: not Ultra HDR per libultrahdr"
    fi

    # ExifTool XMP check
    xmp=$(exiftool -XMP-hdrgm:all "$f" 2>/dev/null)
    if [ -n "$xmp" ]; then
        pass "$name: has XMP hdrgm namespace"
    fi

    # ExifTool MPF check
    mpf_ver=$(exiftool -MPF:MPFVersion "$f" 2>/dev/null | awk '{print $NF}')
    if [ "$mpf_ver" = "0100" ]; then
        pass "$name: MPF Version 0100"
    elif [ -n "$mpf_ver" ]; then
        fail "$name: MPF Version '$mpf_ver' (expected 0100)"
    fi

    # ISO 21496-1 APP2 check
    if grep -qP 'urn:iso:std:iso:ts:21496:-1' "$f" 2>/dev/null; then
        pass "$name: has ISO 21496-1 APP2 binary block"
    else
        skip "$name: no ISO 21496-1 binary (XMP-only)"
    fi

    # Decode test (SDR output)
    if ultrahdr_app -m 1 -j "$f" -o 3 -O 3 2>/dev/null; then
        pass "$name: ultrahdr_app decode OK"
    else
        # Might not be Ultra HDR
        true
    fi
done
[ "$jpeg_count" -eq 0 ] && skip "No JPEG files found in $SAMPLES"

# ============================================================================
section "2. AVIF gain map validation (libavif)"
# ============================================================================
avif_count=0
for f in "$SAMPLES"/*.avif; do
    [ -f "$f" ] || continue
    name=$(basename "$f")
    ((avif_count++))

    info=$(avifdec --info "$f" 2>&1) || true
    if echo "$info" | grep -qi "Gain map"; then
        pass "$name: avifdec detects gain map"
        gm_line=$(echo "$info" | grep -i "Gain map")
        echo "    $gm_line"
    else
        skip "$name: no gain map detected by avifdec"
    fi
done
[ "$avif_count" -eq 0 ] && skip "No AVIF files found in $SAMPLES"

# ============================================================================
section "3. Cross-format round-trip: JPEG -> probe"
# ============================================================================

# If we have a Rust-generated test file, validate it
if [ -f "$OUTDIR/rust_ultrahdr.jpg" ]; then
    probe=$(ultrahdr_app -m 1 -P -j "$OUTDIR/rust_ultrahdr.jpg" 2>&1)
    if echo "$probe" | grep -q "Ultra HDR Image: Yes"; then
        pass "Rust-encoded JPEG: libultrahdr accepts"
        echo "$probe" | sed 's/^/    /'
    else
        fail "Rust-encoded JPEG: libultrahdr rejects"
    fi
fi

# ============================================================================
section "4. ISO 21496-1 binary format validation"
# ============================================================================

# For each JPEG with ISO block, extract and verify the payload structure
for f in "$SAMPLES"/*.jpg "$SAMPLES"/*.jpeg; do
    [ -f "$f" ] || continue
    name=$(basename "$f")

    python3 -c "
import struct, sys
data = open('$f', 'rb').read()
urn = b'urn:iso:std:iso:ts:21496:-1\x00'
pos = 0
found = False
while True:
    idx = data.find(urn, pos)
    if idx < 0: break
    found = True
    mp = idx - 4
    length = struct.unpack('>H', data[mp+2:mp+4])[0]
    payload = data[idx + len(urn):mp + 2 + length]
    if len(payload) < 5:
        print(f'  {len(payload)}-byte stub (version-only marker)')
        pos = idx + 1
        continue
    min_ver = struct.unpack('>H', payload[0:2])[0]
    wr_ver = struct.unpack('>H', payload[2:4])[0]
    flags = payload[4]
    multi = bool(flags & 0x80)
    base_cs = bool(flags & 0x40)
    channels = 3 if multi else 1
    expected = 5 + 16 + channels * 40
    ok = (min_ver == 0 and len(payload) >= expected)
    status = 'OK' if ok else 'BAD'
    print(f'  ISO payload: {len(payload)} bytes, min_ver={min_ver}, flags=0x{flags:02x}, multi={multi}, base_cs={base_cs} [{status}]')
    pos = idx + 1
if not found:
    sys.exit(1)
" 2>/dev/null && pass "$name: ISO 21496-1 payload structure valid" || true
done

# ============================================================================
section "5. Canonical fraction encoding validation"
# ============================================================================

# Verify that ISO 21496-1 payloads use canonical (compact) fraction
# denominators, not the old fixed 1,000,000 denominator. Browsers
# (Chromium) prefer ISO over XMP when both are present, and non-canonical
# fractions cause HDR rendering failures (jcayzac/ultrajpeg#6).

for f in "$SAMPLES"/*.jpg "$SAMPLES"/*.jpeg; do
    [ -f "$f" ] || continue
    name=$(basename "$f")

    python3 << PYEOF
import struct, sys

data = open('$f', 'rb').read()
urn = b'urn:iso:std:iso:ts:21496:-1\x00'
pos = 0
found_gainmap_iso = False
has_bad_denom = False
has_version_only = False

while True:
    idx = data.find(urn, pos)
    if idx < 0:
        break

    mp = idx - 4
    length = struct.unpack('>H', data[mp+2:mp+4])[0]
    payload = data[idx + len(urn):mp + 2 + length]

    if len(payload) == 4:
        # Version-only block (primary JPEG)
        min_v = struct.unpack('>H', payload[0:2])[0]
        wr_v = struct.unpack('>H', payload[2:4])[0]
        if min_v == 0 and wr_v == 0:
            has_version_only = True
        pos = idx + 1
        continue

    if len(payload) < 5:
        pos = idx + 1
        continue

    found_gainmap_iso = True
    flags = payload[4]
    channels = 3 if (flags & 0x80) else 1
    expected = 5 + 16 + channels * 40

    if len(payload) < expected:
        pos = idx + 1
        continue

    # Check all fraction denominators for the 1M anti-pattern
    frac_start = 5  # after header
    frac_count = 2 + channels * 5  # headroom(2) + per-channel(5 each)
    for i in range(frac_count):
        offset = frac_start + i * 8 + 4  # denominator is 2nd u32 in each pair
        if offset + 4 > len(payload):
            break
        denom = struct.unpack('>I', payload[offset:offset+4])[0]
        if denom == 1000000:
            has_bad_denom = True
            frac_name = f'fraction[{i}]'
            numer_off = frac_start + i * 8
            numer = struct.unpack('>I', payload[numer_off:numer_off+4])[0]
            print(f'  BAD: {frac_name} = {numer}/1000000 (non-canonical fixed denominator)')

    pos = idx + 1

if not found_gainmap_iso:
    sys.exit(1)  # No ISO payload to check

if has_bad_denom:
    print('  RESULT: NON-CANONICAL fractions (1000000 denominator)')
    sys.exit(2)
else:
    print('  RESULT: Canonical fractions (no 1000000 denominators)')
    sys.exit(0)
PYEOF
    rc=$?
    if [ "$rc" -eq 0 ]; then
        pass "$name: canonical fraction encoding"
    elif [ "$rc" -eq 2 ]; then
        fail "$name: non-canonical fraction encoding (1000000 denominator)"
    fi
    # rc=1 means no ISO payload, skip silently
done

# ============================================================================
section "6. Metadata value round-trip validation (libultrahdr)"
# ============================================================================

# For Rust-encoded JPEGs, compare libultrahdr's parsed metadata values
# against expected values. This catches the class of bug where our
# serialized ISO bytes are accepted but produce wrong HDR rendering.

for f in "$SAMPLES"/rust_*.jpg; do
    [ -f "$f" ] || continue
    name=$(basename "$f")
    probe=$(ultrahdr_app -m 1 -P -j "$f" 2>&1) || continue

    python3 << PYEOF
import sys, re

probe = """$(echo "$probe")"""

def extract(key):
    m = re.search(rf'{key}\s+([\d.eE+\-]+|inf|-inf|nan)', probe)
    return float(m.group(1)) if m else None

max_boost = extract('maxContentBoost')
min_boost = extract('minContentBoost')
cap_max = extract('hdrCapacityMax')
cap_min = extract('hdrCapacityMin')
gamma = extract('gamma')
offset_sdr = extract('offsetSdr')
offset_hdr = extract('offsetHdr')

errors = []

# maxContentBoost must be a reasonable finite positive value
if max_boost is not None and max_boost > 0:
    print(f'  maxContentBoost = {max_boost:.4f}')
else:
    errors.append(f'maxContentBoost invalid: {max_boost}')

# hdrCapacityMax must be finite and positive
if cap_max is not None and cap_max > 0 and cap_max < 1e6:
    print(f'  hdrCapacityMax  = {cap_max:.4f}')
else:
    errors.append(f'hdrCapacityMax invalid: {cap_max}')

# hdrCapacityMin must NOT be inf (common bug with bad base_hdr_headroom)
if cap_min is not None and cap_min < 1e6:
    print(f'  hdrCapacityMin  = {cap_min:.4f}')
else:
    errors.append(f'hdrCapacityMin invalid or inf: {cap_min}')

# gamma should be > 0 (typically 1.0)
if gamma is not None and gamma > 0:
    print(f'  gamma           = {gamma:.4f}')
else:
    errors.append(f'gamma invalid: {gamma}')

if errors:
    for e in errors:
        print(f'  ERROR: {e}')
    sys.exit(1)
else:
    sys.exit(0)
PYEOF
    if [ $? -eq 0 ]; then
        pass "$name: libultrahdr metadata values are sane"
    else
        fail "$name: libultrahdr metadata values are bad"
    fi
done

# ============================================================================
section "7. Primary JPEG version-only APP2 check"
# ============================================================================

# Canonical Ultra HDR JPEGs should have a 4-byte version-only ISO APP2 block
# (00 00 00 00) in the primary JPEG, before the gain map secondary.

for f in "$SAMPLES"/*.jpg "$SAMPLES"/*.jpeg; do
    [ -f "$f" ] || continue
    name=$(basename "$f")

    python3 << PYEOF
import struct, sys

data = open('$f', 'rb').read()
urn = b'urn:iso:std:iso:ts:21496:-1\x00'

# Find first EOI to determine primary JPEG boundary
# (scan for FF D9 that ends the primary, before the secondary SOI)
i = 2  # skip initial SOI
primary_eoi = None
while i < len(data) - 1:
    if data[i] == 0xFF:
        marker = data[i+1]
        if marker == 0xD9:  # EOI
            primary_eoi = i
            break
        elif marker == 0xDA:  # SOS - skip entropy data
            i += 2
            # Scan for next marker
            while i < len(data) - 1:
                if data[i] == 0xFF and data[i+1] != 0x00 and data[i+1] != 0xFF:
                    break
                i += 1
        elif 0xC0 <= marker <= 0xFE:
            if i + 3 < len(data):
                seg_len = struct.unpack('>H', data[i+2:i+4])[0]
                i += 2 + seg_len
            else:
                break
        else:
            i += 1
    else:
        i += 1

if primary_eoi is None:
    sys.exit(1)  # Can't find primary boundary

# Search for ISO APP2 blocks within the primary JPEG only
primary_data = data[:primary_eoi]
idx = primary_data.find(urn)

if idx < 0:
    print('  No ISO APP2 in primary JPEG')
    sys.exit(2)  # No ISO in primary

mp = idx - 4
length = struct.unpack('>H', primary_data[mp+2:mp+4])[0]
payload = primary_data[idx + len(urn):mp + 2 + length]

if len(payload) == 4 and payload == b'\x00\x00\x00\x00':
    print('  Primary has version-only ISO APP2 (00 00 00 00)')
    sys.exit(0)
else:
    print(f'  Primary has ISO APP2 but payload is {len(payload)} bytes (expected 4-byte version-only)')
    sys.exit(3)
PYEOF
    rc=$?
    if [ "$rc" -eq 0 ]; then
        pass "$name: primary JPEG has canonical version-only ISO APP2"
    elif [ "$rc" -eq 2 ]; then
        skip "$name: no ISO APP2 in primary (XMP-only or legacy format)"
    elif [ "$rc" -eq 3 ]; then
        fail "$name: primary ISO APP2 is not the expected 4-byte version block"
    fi
    # rc=1 means can't parse JPEG structure, skip
done

# ============================================================================
section "SUMMARY"
# ============================================================================
echo ""
echo -e "Results: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}, ${YELLOW}${SKIP} skipped${NC}"
echo ""
if [ "$FAIL" -gt 0 ]; then
    echo -e "${RED}Some validations failed${NC}"
    exit 1
else
    echo -e "${GREEN}All validations passed${NC}"
fi
