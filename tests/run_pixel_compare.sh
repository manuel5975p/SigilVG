#!/bin/bash
# Pixel-compare SigilVG renders against resvg reference PNGs.
# Uses ImageMagick 'compare' for PSNR and absolute error metrics.

HEADLESS="${1:-./sigilvg_headless}"
SUITE_DIR="${2:-../tests/resvg_suite/tests}"
OUT_DIR="${3:-pixel_results}"
TIMEOUT_SEC=10

mkdir -p "$OUT_DIR"

LOG="$OUT_DIR/pixel_results.csv"
echo "svg,category,subcategory,ref_w,ref_h,status,psnr,mae,pct_diff" > "$LOG"

total=0
render_ok=0
skip=0
fail=0
good=0    # PSNR >= 20
great=0   # PSNR >= 30
perfect=0 # PSNR >= 40

find "$SUITE_DIR" -name "*.svg" -type f | sort | while read -r svgpath; do
    rel="${svgpath#$SUITE_DIR/}"
    category=$(echo "$rel" | cut -d/ -f1)
    subcategory=$(echo "$rel" | cut -d/ -f2)
    name=$(basename "$svgpath" .svg)
    refpng="${svgpath%.svg}.png"

    # Skip if no reference PNG
    if [ ! -f "$refpng" ]; then
        echo "$rel,$category,$subcategory,0,0,NO_REF,,,," >> "$LOG"
        skip=$((skip + 1))
        total=$((total + 1))
        continue
    fi

    # Get reference dimensions
    dims=$(identify -format "%w %h" "$refpng" 2>/dev/null)
    ref_w=$(echo "$dims" | awk '{print $1}')
    ref_h=$(echo "$dims" | awk '{print $2}')
    if [ -z "$ref_w" ] || [ "$ref_w" -eq 0 ]; then
        echo "$rel,$category,$subcategory,0,0,BAD_REF,,,," >> "$LOG"
        skip=$((skip + 1))
        total=$((total + 1))
        continue
    fi

    outpng="$OUT_DIR/render_${category}_${subcategory}_${name}.png"

    # Render at reference dimensions
    stderr_out=$(timeout "$TIMEOUT_SEC" "$HEADLESS" "$svgpath" "$outpng" "$ref_w" "$ref_h" 2>&1)
    ec=$?

    total=$((total + 1))

    if [ $ec -ne 0 ] || [ ! -f "$outpng" ]; then
        echo "$rel,$category,$subcategory,$ref_w,$ref_h,RENDER_FAIL,,,," >> "$LOG"
        fail=$((fail + 1))
        if [ $((total % 50)) -eq 0 ]; then
            echo "[$total] good=$good great=$great perfect=$perfect fail=$fail skip=$skip" >&2
        fi
        continue
    fi

    render_ok=$((render_ok + 1))

    # Flatten reference PNG onto white (resvg refs have transparent bg, SigilVG uses white)
    flatref="$OUT_DIR/_flat_ref.png"
    magick "$refpng" -background white -flatten "$flatref" 2>/dev/null

    # Compare with ImageMagick — get PSNR
    cmp_out=$(compare -metric PSNR "$outpng" "$flatref" /dev/null 2>&1)
    psnr=$(echo "$cmp_out" | grep -oP '[\d.]+' | head -1)
    # If images are identical, compare outputs "inf"
    if echo "$cmp_out" | grep -qi "inf"; then
        psnr="999"
    fi

    # Get percentage of differing pixels (AE = absolute error count)
    ae_out=$(compare -metric AE -fuzz 5% "$outpng" "$flatref" /dev/null 2>&1)
    ae_count=$(echo "$ae_out" | grep -oP '[\d.]+' | head -1)
    total_px=$((ref_w * ref_h))
    if [ "$total_px" -gt 0 ] && [ -n "$ae_count" ]; then
        pct_diff=$(awk "BEGIN {printf \"%.2f\", $ae_count / $total_px * 100}")
    else
        pct_diff="?"
    fi

    # Classify
    if [ -n "$psnr" ]; then
        is_good=$(awk "BEGIN {print ($psnr >= 20) ? 1 : 0}")
        is_great=$(awk "BEGIN {print ($psnr >= 30) ? 1 : 0}")
        is_perfect=$(awk "BEGIN {print ($psnr >= 40) ? 1 : 0}")
        [ "$is_good" -eq 1 ] && good=$((good + 1))
        [ "$is_great" -eq 1 ] && great=$((great + 1))
        [ "$is_perfect" -eq 1 ] && perfect=$((perfect + 1))
    fi

    echo "$rel,$category,$subcategory,$ref_w,$ref_h,OK,$psnr,,$pct_diff" >> "$LOG"

    # Clean up rendered PNG and temp files
    rm -f "$outpng" "$flatref"

    if [ $((total % 50)) -eq 0 ]; then
        echo "[$total] good=$good great=$great perfect=$perfect fail=$fail skip=$skip" >&2
    fi
done

# Final summary from CSV
total=$(tail -n +2 "$LOG" | wc -l)
render_ok=$(grep -c ",OK," "$LOG")
render_fail=$(grep -c ",RENDER_FAIL," "$LOG")
no_ref=$(grep -c ",NO_REF," "$LOG")

# Extract PSNR values for stats
psnr_vals=$(grep ",OK," "$LOG" | awk -F, '{print $7}' | grep -v '^$')
good=$(echo "$psnr_vals" | awk '$1 >= 20' | wc -l)
great=$(echo "$psnr_vals" | awk '$1 >= 30' | wc -l)
perfect=$(echo "$psnr_vals" | awk '$1 >= 40' | wc -l)
identical=$(echo "$psnr_vals" | awk '$1 >= 999' | wc -l)
below20=$(echo "$psnr_vals" | awk '$1 < 20' | wc -l)

median_psnr=$(echo "$psnr_vals" | sort -n | awk 'NR==int(NR/2)+1{print}')
mean_psnr=$(echo "$psnr_vals" | awk '{s+=$1; n++} END {if(n>0) printf "%.1f", s/n}')

echo ""
echo "=== SigilVG Pixel Comparison Results ==="
echo "Total SVGs tested:  $total"
echo "Rendered OK:        $render_ok"
echo "Render failed:      $render_fail"
echo "No reference PNG:   $no_ref"
echo ""
echo "=== PSNR Distribution (of $render_ok rendered) ==="
echo "  Identical (inf):  $identical"
echo "  >= 40 dB (great): $perfect"
echo "  >= 30 dB (good):  $great"
echo "  >= 20 dB (ok):    $good"
echo "  < 20 dB (poor):   $below20"
echo ""
echo "  Mean PSNR:   $mean_psnr dB"
echo "  Median PSNR: $median_psnr dB"
echo ""

# Per-category
echo "=== Per-Category Mean PSNR ==="
for cat in filters masking painting paint-servers shapes structure text; do
    cat_data=$(grep ",$cat," "$LOG" | grep ",OK," | awk -F, '{print $7}' | grep -v '^$')
    cat_n=$(echo "$cat_data" | grep -c .)
    cat_mean=$(echo "$cat_data" | awk '{s+=$1; n++} END {if(n>0) printf "%.1f", s/n}')
    cat_below20=$(echo "$cat_data" | awk '$1 < 20' | wc -l)
    if [ "$cat_n" -gt 0 ]; then
        echo "  $cat: mean=${cat_mean}dB, n=$cat_n, poor(<20dB)=$cat_below20"
    fi
done

echo ""
echo "=== Worst 20 SVGs by PSNR ==="
grep ",OK," "$LOG" | awk -F, '{print $7, $1}' | grep -v '^$' | sort -n | head -20

echo ""
echo "Full results: $LOG"
