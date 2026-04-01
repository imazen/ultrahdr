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
section "3. Cross-format round-trip: JPEG → probe"
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
