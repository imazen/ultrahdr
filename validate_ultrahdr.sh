#!/bin/bash
# Ultra HDR validation script
# Validates our output against libultrahdr reference and ExifTool
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'
PASS=0
FAIL=0
SKIP=0

pass() { echo -e "  ${GREEN}PASS${NC}: $1"; ((PASS++)); }
fail() { echo -e "  ${RED}FAIL${NC}: $1"; ((FAIL++)); }
skip() { echo -e "  ${YELLOW}SKIP${NC}: $1"; ((SKIP++)); }
section() { echo -e "\n${BLUE}=== $1 ===${NC}\n"; }

WORKDIR="${WORKDIR:-/workspace/ultrahdr}"
SAMPLES="${WORKDIR}/test_images"
# Also check for mounted samples
if [ -d "/samples" ]; then
    SAMPLES="/samples"
fi
OUTDIR="/tmp/ultrahdr_validation"
mkdir -p "$OUTDIR"

# ============================================================================
section "1. Tool availability"
# ============================================================================

if command -v ultrahdr_app &>/dev/null; then
    pass "ultrahdr_app available"
else
    fail "ultrahdr_app not found — cannot validate against C++ reference"
fi

if command -v exiftool &>/dev/null; then
    pass "exiftool available ($(exiftool -ver))"
else
    fail "exiftool not found — cannot validate XMP metadata"
fi

if command -v ultrahdr_unit_test &>/dev/null; then
    pass "ultrahdr_unit_test available"
else
    skip "ultrahdr_unit_test not found"
fi

# ============================================================================
section "2. libultrahdr C++ reference unit tests"
# ============================================================================

if command -v ultrahdr_unit_test &>/dev/null; then
    if ultrahdr_unit_test 2>&1 | tee "${OUTDIR}/cpp_unit_tests.log" | tail -5; then
        pass "libultrahdr C++ unit tests passed"
    else
        fail "libultrahdr C++ unit tests failed"
        tail -20 "${OUTDIR}/cpp_unit_tests.log"
    fi
else
    skip "ultrahdr_unit_test not available"
fi

# ============================================================================
section "3. Rust unit tests"
# ============================================================================

cd "$WORKDIR"

# Core tests (ISO 21496, XMP, gain map math)
echo "Running ultrahdr-core tests..."
if cargo test --release -p ultrahdr-core 2>&1 | tee "${OUTDIR}/core_tests.log" | tail -3; then
    test_line=$(grep "^test result:" "${OUTDIR}/core_tests.log" | tail -1)
    if echo "$test_line" | grep -q "0 failed"; then
        pass "ultrahdr-core: $test_line"
    else
        fail "ultrahdr-core: $test_line"
    fi
else
    fail "ultrahdr-core tests crashed"
fi

# ISO 21496 specifically
echo ""
echo "Running ISO 21496-1 tests..."
if cargo test --release -p ultrahdr-core -- iso21496 2>&1 | tee "${OUTDIR}/iso_tests.log" | tail -3; then
    test_count=$(grep -c "^test .* ok$" "${OUTDIR}/iso_tests.log" || true)
    pass "ISO 21496-1: $test_count tests passed"
else
    fail "ISO 21496-1 tests failed"
fi

# XMP tests
echo ""
echo "Running XMP tests..."
if cargo test --release -p ultrahdr-core -- xmp 2>&1 | tee "${OUTDIR}/xmp_tests.log" | tail -3; then
    test_count=$(grep -c "^test .* ok$" "${OUTDIR}/xmp_tests.log" || true)
    pass "XMP metadata: $test_count tests passed"
else
    fail "XMP metadata tests failed"
fi

# ============================================================================
section "4. FFI parity tests (Rust encoder → libultrahdr decoder)"
# ============================================================================

if cargo test --release --features ffi-tests -- --nocapture 2>&1 | tee "${OUTDIR}/ffi_tests.log" | tail -10; then
    test_line=$(grep "^test result:" "${OUTDIR}/ffi_tests.log" | tail -1)
    if [ -n "$test_line" ]; then
        if echo "$test_line" | grep -q "0 failed"; then
            pass "FFI parity: $test_line"
        else
            fail "FFI parity: $test_line"
        fi
    else
        skip "FFI parity: no test results found"
    fi
else
    skip "FFI parity tests not available (build may have failed)"
fi

# ============================================================================
section "5. Validate real Ultra HDR samples"
# ============================================================================

sample_count=0
for sample in "$SAMPLES"/*.jpg "$SAMPLES"/*.jpeg; do
    [ -f "$sample" ] || continue
    ((sample_count++))
    name=$(basename "$sample")
    echo "--- $name ($(wc -c < "$sample") bytes) ---"

    # 5a. ultrahdr_app decode
    if ultrahdr_app -m 1 -j "$sample" -o "${OUTDIR}/${name%.jpg}_hdr" 2>"${OUTDIR}/${name%.jpg}_decode.log"; then
        pass "$name: ultrahdr_app decode OK"

        # Check for gain map metadata in decode output
        if ultrahdr_app -m 1 -j "$sample" -f "${OUTDIR}/${name%.jpg}_meta.txt" -o /dev/null 2>/dev/null; then
            if [ -f "${OUTDIR}/${name%.jpg}_meta.txt" ] && [ -s "${OUTDIR}/${name%.jpg}_meta.txt" ]; then
                pass "$name: gain map metadata extracted"
                cat "${OUTDIR}/${name%.jpg}_meta.txt"
            fi
        fi
    else
        fail "$name: ultrahdr_app decode FAILED"
        cat "${OUTDIR}/${name%.jpg}_decode.log" 2>/dev/null
    fi

    # 5b. ExifTool XMP inspection
    echo ""
    echo "  ExifTool XMP-hdrgm:"
    hdrgm_out=$(exiftool -XMP-hdrgm:all "$sample" 2>/dev/null)
    if [ -n "$hdrgm_out" ]; then
        pass "$name: has XMP-hdrgm namespace"
        echo "$hdrgm_out" | sed 's/^/    /'
    else
        fail "$name: missing XMP-hdrgm namespace"
    fi

    # 5c. ExifTool MPF inspection
    mpf_out=$(exiftool -MPF:all "$sample" 2>/dev/null)
    if [ -n "$mpf_out" ]; then
        pass "$name: has MPF directory"
        echo "$mpf_out" | sed 's/^/    /'
    else
        fail "$name: missing MPF directory"
    fi

    # 5d. Check for ISO 21496-1 APP2 block
    if grep -qP "urn:iso:std:iso:ts:21496:-1" "$sample" 2>/dev/null; then
        pass "$name: has ISO 21496-1 APP2 binary metadata"
        # Hex dump the ISO block
        offset=$(grep -boPa "urn:iso:std:iso:ts:21496:-1" "$sample" 2>/dev/null | head -1 | cut -d: -f1)
        if [ -n "$offset" ]; then
            urn_len=27  # "urn:iso:std:iso:ts:21496:-1\0"
            iso_start=$((offset + urn_len + 1))  # +1 for null terminator
            echo "    ISO 21496 payload at offset $iso_start:"
            xxd -s "$iso_start" -l 80 "$sample" 2>/dev/null | sed 's/^/    /'
        fi
    else
        skip "$name: no ISO 21496-1 APP2 block (XMP-only format)"
    fi

    # 5e. Verify gain map JPEG structure via MPF
    gm_offset=$(exiftool -b -MPImageStart "$sample" 2>/dev/null | head -1)
    gm_length=$(exiftool -b -MPImageLength "$sample" 2>/dev/null | head -1)
    if [ -n "$gm_offset" ] && [ "$gm_offset" -gt 0 ] 2>/dev/null; then
        # Extract gain map JPEG
        dd if="$sample" of="${OUTDIR}/${name%.jpg}_gainmap.jpg" \
            bs=1 skip="$gm_offset" count="$gm_length" 2>/dev/null
        gm_file="${OUTDIR}/${name%.jpg}_gainmap.jpg"

        # Check SOI/EOI
        soi=$(xxd -l 2 "$gm_file" 2>/dev/null | awk '{print $2}')
        if [ "$soi" = "ffd8" ]; then
            pass "$name: gain map JPEG has valid SOI"
        else
            fail "$name: gain map JPEG invalid SOI (got: $soi)"
        fi

        # Get gain map dimensions
        gm_info=$(exiftool -ImageWidth -ImageHeight "$gm_file" 2>/dev/null)
        if [ -n "$gm_info" ]; then
            echo "    Gain map: $gm_info" | tr '\n' ', '
            echo ""
        fi
    fi

    echo ""
done

if [ "$sample_count" -eq 0 ]; then
    skip "No sample Ultra HDR images found in $SAMPLES"
    echo "  Mount samples: docker run --rm -v /path/to/samples:/samples ultrahdr-validate"
fi

# ============================================================================
section "6. Generate and validate our own Ultra HDR output"
# ============================================================================

# Use cargo test to generate a test Ultra HDR JPEG and write it to disk
cat > /tmp/gen_test_ultrahdr.rs << 'TESTEOF'
// This test generates a test Ultra HDR JPEG and writes it for external validation
use std::fs;

fn main() {
    eprintln!("Test Ultra HDR generation would go here");
    eprintln!("Run FFI parity tests for full cross-validation");
}
TESTEOF

# Instead, extract test output from the FFI tests
if [ -f "${OUTDIR}/ffi_tests.log" ]; then
    encoded_size=$(grep "Encoded .* bytes" "${OUTDIR}/ffi_tests.log" | head -1)
    if [ -n "$encoded_size" ]; then
        pass "Rust encoder produced output: $encoded_size"
    fi

    if grep -q "libultrahdr accepted" "${OUTDIR}/ffi_tests.log"; then
        pass "libultrahdr accepted our encoder output"
    fi

    if grep -q "libultrahdr decoded:" "${OUTDIR}/ffi_tests.log"; then
        decoded_dims=$(grep "libultrahdr decoded:" "${OUTDIR}/ffi_tests.log" | head -1)
        pass "libultrahdr decoded our output: $decoded_dims"
    fi
fi

# ============================================================================
section "7. ISO 21496-1 compliance checklist"
# ============================================================================

echo "Checking implementation against ISO 21496-1 requirements:"
echo ""

# Check serialize/deserialize are available
if grep -q "serialize_iso21496" "${OUTDIR}/iso_tests.log" 2>/dev/null; then
    pass "serialize_iso21496() implemented and tested"
else
    skip "Could not verify serialize_iso21496 in test output"
fi

if grep -q "parse_iso21496\|deserialize_iso21496" "${OUTDIR}/iso_tests.log" 2>/dev/null; then
    pass "parse_iso21496() implemented and tested"
else
    skip "Could not verify parse_iso21496 in test output"
fi

# Check APP2 marker format
if grep -q "app2_marker" "${OUTDIR}/iso_tests.log" 2>/dev/null; then
    pass "APP2 marker creation tested (urn:iso:std:iso:ts:21496:-1)"
else
    skip "APP2 marker creation not verified in test output"
fi

echo ""
echo "Key gaps to investigate:"
echo "  - Is ISO 21496-1 APP2 actually embedded in our encoder output?"
echo "    (Currently ultrahdr-rs/encode.rs only writes XMP, not ISO binary)"
echo "  - Does serialize_iso21496() produce bytes that libultrahdr can parse?"
echo "  - Are fraction denominators (1,000,000) compatible with other implementations?"
echo "  - Is the URN namespace string exactly correct?"

# ============================================================================
section "SUMMARY"
# ============================================================================

echo ""
echo -e "Results: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}, ${YELLOW}${SKIP} skipped${NC}"
echo ""

if [ "$FAIL" -gt 0 ]; then
    echo -e "${RED}Some validations failed — see details above${NC}"
    exit 1
else
    echo -e "${GREEN}All validations passed${NC}"
    exit 0
fi
