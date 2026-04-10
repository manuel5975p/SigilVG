#!/bin/bash
# Batch-run SigilVG headless renderer against resvg test suite.
# Categorizes each SVG as: RENDER (produced PNG), PARSE_FAIL (no elements), CRASH (segfault/abort), TIMEOUT.

HEADLESS="${1:-./sigilvg_headless}"
SUITE_DIR="${2:-../tests/resvg_suite/tests}"
OUT_DIR="${3:-resvg_results}"
TIMEOUT_SEC=5

mkdir -p "$OUT_DIR"

total=0
render_ok=0
parse_fail=0
crash=0
timeout_count=0

# CSV log
LOG="$OUT_DIR/results.csv"
echo "svg,category,subcategory,status,exit_code,stderr" > "$LOG"

find "$SUITE_DIR" -name "*.svg" -type f | sort | while read -r svgpath; do
    rel="${svgpath#$SUITE_DIR/}"
    category=$(echo "$rel" | cut -d/ -f1)
    subcategory=$(echo "$rel" | cut -d/ -f2)
    name=$(basename "$svgpath" .svg)

    outpng="$OUT_DIR/${category}_${subcategory}_${name}.png"

    # Run with timeout
    stderr_out=$(timeout "$TIMEOUT_SEC" "$HEADLESS" "$svgpath" "$outpng" 2>&1)
    ec=$?

    total=$((total + 1))

    if [ $ec -eq 124 ]; then
        status="TIMEOUT"
        timeout_count=$((timeout_count + 1))
    elif [ $ec -ne 0 ]; then
        if echo "$stderr_out" | grep -q "no renderable elements"; then
            status="PARSE_FAIL"
            parse_fail=$((parse_fail + 1))
        else
            status="CRASH"
            crash=$((crash + 1))
        fi
    else
        if [ -f "$outpng" ]; then
            status="RENDER_OK"
            render_ok=$((render_ok + 1))
            # Clean up PNGs to save space - we only care about stats
            rm -f "$outpng"
        else
            status="NO_OUTPUT"
            crash=$((crash + 1))
        fi
    fi

    # Truncate stderr for CSV
    stderr_short=$(echo "$stderr_out" | head -1 | tr ',' ';' | cut -c1-100)
    echo "$rel,$category,$subcategory,$status,$ec,$stderr_short" >> "$LOG"

    # Progress every 50
    if [ $((total % 50)) -eq 0 ]; then
        echo "[$total] ok=$render_ok parse_fail=$parse_fail crash=$crash timeout=$timeout_count" >&2
    fi
done

# Summary (read back from CSV since subshell)
total=$(tail -n +2 "$LOG" | wc -l)
render_ok=$(grep -c ",RENDER_OK," "$LOG")
parse_fail=$(grep -c ",PARSE_FAIL," "$LOG")
crash=$(grep -c ",CRASH," "$LOG")
timeout_count=$(grep -c ",TIMEOUT," "$LOG")
no_output=$(grep -c ",NO_OUTPUT," "$LOG")

echo ""
echo "=== SigilVG resvg Suite Results ==="
echo "Total SVGs:   $total"
echo "Render OK:    $render_ok  ($(( render_ok * 100 / total ))%)"
echo "Parse fail:   $parse_fail  ($(( parse_fail * 100 / total ))%)"
echo "Crash/error:  $crash  ($(( crash * 100 / total ))%)"
echo "Timeout:      $timeout_count  ($(( timeout_count * 100 / total ))%)"
echo "No output:    $no_output"
echo ""

# Per-category breakdown
echo "=== Per-Category Breakdown ==="
for cat in filters masking painting paint-servers shapes structure text; do
    cat_total=$(grep ",$cat," "$LOG" | wc -l)
    cat_ok=$(grep ",$cat," "$LOG" | grep -c ",RENDER_OK,")
    if [ "$cat_total" -gt 0 ]; then
        pct=$(( cat_ok * 100 / cat_total ))
        echo "  $cat: $cat_ok / $cat_total ($pct%)"
    fi
done

echo ""
echo "Details in: $LOG"
