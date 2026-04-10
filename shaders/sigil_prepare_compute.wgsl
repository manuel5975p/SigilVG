// ===================================================
// GPU-driven prepare compute shader for SigilVG
// Replaces the CPU sigil_prepare() path.
// Dispatched as dispatch(element_count, 1, 1), workgroup size 1.
// Each invocation processes one SVG element.
// ===================================================

const BAND_COUNT: u32 = 8u;

// --------------- Input structs ---------------

struct Curve {
    p0x : f32, p0y : f32,
    p1x : f32, p1y : f32,
    p2x : f32, p2y : f32,
};

struct ElementMeta {
    curve_offset  : u32,
    curve_count   : u32,
    bounds_xMin   : f32, bounds_yMin : f32,
    bounds_xMax   : f32, bounds_yMax : f32,
    fill_r : f32, fill_g : f32, fill_b : f32, fill_a : f32,
    fill_rule     : u32,
    gradient_idx  : i32,
    opacity       : f32,
    pad           : u32,
    grad0_x : f32, grad0_y : f32, grad0_z : f32, grad0_w : f32,
    grad1_x : f32, grad1_y : f32, grad1_z : f32, grad1_w : f32,
};

struct ElementOffset {
    curve_start : u32,
    band_start  : u32,
};

struct Viewport {
    width     : f32, height    : f32,
    vb_x      : f32, vb_y      : f32,
    vb_w      : f32, vb_h      : f32,
    scale     : f32, inv_scale : f32,
};

// --------------- Bindings ---------------

// group(0): inputs
@group(0) @binding(0) var<storage, read> curves   : array<Curve>;
@group(0) @binding(1) var<storage, read> elements : array<ElementMeta>;
@group(0) @binding(2) var<storage, read> offsets  : array<ElementOffset>;
@group(0) @binding(3) var<storage, read> gradDefs : array<u32>;
@group(0) @binding(4) var<uniform>       viewport : Viewport;

// group(1): outputs
@group(1) @binding(0) var<storage, read_write> curveBuf  : array<vec4<f32>>;
@group(1) @binding(1) var<storage, read_write> bandBuf   : array<vec4<u32>>;
@group(1) @binding(2) var<storage, read_write> vertexBuf : array<f32>;
@group(1) @binding(3) var<storage, read_write> indexBuf  : array<u32>;

// --------------- Entry point ---------------

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let ei = gid.x;
    let elem = elements[ei];
    let off  = offsets[ei];
    let nc   = elem.curve_count;

    // Element bounds
    let xMin = elem.bounds_xMin;
    let yMin = elem.bounds_yMin;
    let xMax = elem.bounds_xMax;
    let yMax = elem.bounds_yMax;
    let w    = xMax - xMin;
    let h    = yMax - yMin;

    // Band scale / offset
    let bsX = select(0.0, 8.0 / w, w > 0.0);
    let bsY = select(0.0, 8.0 / h, h > 0.0);
    let boX = -xMin * bsX;
    let boY = -yMin * bsY;

    // ---------------------------------------------------------------
    // Step 1: Pack curves into curveBuf
    // ---------------------------------------------------------------
    // Each curve i occupies two vec4s at curveBuf[off.curve_start + i*2 + 0..1].
    for (var ci = 0u; ci < nc; ci++) {
        let c = curves[elem.curve_offset + ci];
        let base = off.curve_start + ci * 2u;
        curveBuf[base]      = vec4<f32>(c.p0x, c.p0y, c.p1x, c.p1y);
        curveBuf[base + 1u] = vec4<f32>(c.p2x, c.p2y, 0.0, 0.0);
    }

    // ---------------------------------------------------------------
    // Step 2: Build bands
    // ---------------------------------------------------------------
    // Band data layout in bandBuf starting at off.band_start:
    //   [0..15]                         = 16 header entries (written in step 4)
    //   [16 + b*nc .. 16 + b*nc + ...]  = hband b curve refs (b=0..7)
    //   [16 + (8+b)*nc .. ]             = vband b curve refs (b=0..7)
    //
    // We use local counters for how many curves are in each band.

    // Explicit band counters (avoids var array<> in simple_wgsl)
    var h0 = 0u; var h1 = 0u; var h2 = 0u; var h3 = 0u;
    var h4 = 0u; var h5 = 0u; var h6 = 0u; var h7 = 0u;
    var v0 = 0u; var v1 = 0u; var v2 = 0u; var v3 = 0u;
    var v4 = 0u; var v5 = 0u; var v6 = 0u; var v7 = 0u;

    let bandDataBase = off.band_start + 16u; // past the 16 headers

    for (var ci = 0u; ci < nc; ci++) {
        let c = curves[elem.curve_offset + ci];

        // Control-point bounding box
        let cxMin = min(min(c.p0x, c.p1x), c.p2x);
        let cxMax = max(max(c.p0x, c.p1x), c.p2x);
        let cyMin = min(min(c.p0y, c.p1y), c.p2y);
        let cyMax = max(max(c.p0y, c.p1y), c.p2y);

        let curveRef = off.curve_start + ci * 2u;

        // Horizontal bands (partition by Y)
        if (h > 0.0) {
            let b0 = u32(clamp(floor(cyMin * bsY + boY), 0.0, 7.0));
            let b1 = u32(clamp(floor(cyMax * bsY + boY), 0.0, 7.0));
            for (var b = b0; b <= b1; b++) {
                var hc = 0u;
                if (b == 0u) { hc = h0; } else if (b == 1u) { hc = h1; }
                else if (b == 2u) { hc = h2; } else if (b == 3u) { hc = h3; }
                else if (b == 4u) { hc = h4; } else if (b == 5u) { hc = h5; }
                else if (b == 6u) { hc = h6; } else { hc = h7; }
                let slot = bandDataBase + b * nc + hc;
                bandBuf[slot] = vec4<u32>(curveRef, 0u, 0u, 0u);
                hc++;
                if (b == 0u) { h0 = hc; } else if (b == 1u) { h1 = hc; }
                else if (b == 2u) { h2 = hc; } else if (b == 3u) { h3 = hc; }
                else if (b == 4u) { h4 = hc; } else if (b == 5u) { h5 = hc; }
                else if (b == 6u) { h6 = hc; } else { h7 = hc; }
            }
        }

        // Vertical bands (partition by X)
        if (w > 0.0) {
            let b0 = u32(clamp(floor(cxMin * bsX + boX), 0.0, 7.0));
            let b1 = u32(clamp(floor(cxMax * bsX + boX), 0.0, 7.0));
            for (var b = b0; b <= b1; b++) {
                var vc = 0u;
                if (b == 0u) { vc = v0; } else if (b == 1u) { vc = v1; }
                else if (b == 2u) { vc = v2; } else if (b == 3u) { vc = v3; }
                else if (b == 4u) { vc = v4; } else if (b == 5u) { vc = v5; }
                else if (b == 6u) { vc = v6; } else { vc = v7; }
                let slot = bandDataBase + (BAND_COUNT + b) * nc + vc;
                bandBuf[slot] = vec4<u32>(curveRef, 0u, 0u, 0u);
                vc++;
                if (b == 0u) { v0 = vc; } else if (b == 1u) { v1 = vc; }
                else if (b == 2u) { v2 = vc; } else if (b == 3u) { v3 = vc; }
                else if (b == 4u) { v4 = vc; } else if (b == 5u) { v5 = vc; }
                else if (b == 6u) { v6 = vc; } else { v7 = vc; }
            }
        }
    }

    // ---------------------------------------------------------------
    // Step 3: Sort bands (insertion sort, descending by max coord)
    // ---------------------------------------------------------------
    // hbands: sort by descending max-x of referenced curve
    // vbands: sort by descending max-y of referenced curve

    for (var b = 0u; b < BAND_COUNT; b++) {
        var cnt = 0u;
        if (b == 0u) { cnt = h0; } else if (b == 1u) { cnt = h1; }
        else if (b == 2u) { cnt = h2; } else if (b == 3u) { cnt = h3; }
        else if (b == 4u) { cnt = h4; } else if (b == 5u) { cnt = h5; }
        else if (b == 6u) { cnt = h6; } else { cnt = h7; }
        let base = bandDataBase + b * nc;
        // Insertion sort descending by max-x
        for (var i = 1u; i < cnt; i++) {
            let keyRef = bandBuf[base + i].x;
            let keyVec = bandBuf[base + i];
            // Max x of the referenced curve: read p0x,p1x from curveBuf[ref], p2x from curveBuf[ref+1]
            let cv0 = curveBuf[keyRef];
            let cv1 = curveBuf[keyRef + 1u];
            let keyVal = max(max(cv0.x, cv0.z), cv1.x);
            var j = i;
            loop {
                if (j == 0u) { break; }
                let prevRef = bandBuf[base + j - 1u].x;
                let pv0 = curveBuf[prevRef];
                let pv1 = curveBuf[prevRef + 1u];
                let prevVal = max(max(pv0.x, pv0.z), pv1.x);
                if (prevVal >= keyVal) { break; }
                bandBuf[base + j] = bandBuf[base + j - 1u];
                j--;
            }
            bandBuf[base + j] = keyVec;
        }
    }

    for (var b = 0u; b < BAND_COUNT; b++) {
        var cnt = 0u;
        if (b == 0u) { cnt = v0; } else if (b == 1u) { cnt = v1; }
        else if (b == 2u) { cnt = v2; } else if (b == 3u) { cnt = v3; }
        else if (b == 4u) { cnt = v4; } else if (b == 5u) { cnt = v5; }
        else if (b == 6u) { cnt = v6; } else { cnt = v7; }
        let base = bandDataBase + (BAND_COUNT + b) * nc;
        // Insertion sort descending by max-y
        for (var i = 1u; i < cnt; i++) {
            let keyRef = bandBuf[base + i].x;
            let keyVec = bandBuf[base + i];
            let cv0 = curveBuf[keyRef];
            let cv1 = curveBuf[keyRef + 1u];
            let keyVal = max(max(cv0.y, cv0.w), cv1.y);
            var j = i;
            loop {
                if (j == 0u) { break; }
                let prevRef = bandBuf[base + j - 1u].x;
                let pv0 = curveBuf[prevRef];
                let pv1 = curveBuf[prevRef + 1u];
                let prevVal = max(max(pv0.y, pv0.w), pv1.y);
                if (prevVal >= keyVal) { break; }
                bandBuf[base + j] = bandBuf[base + j - 1u];
                j--;
            }
            bandBuf[base + j] = keyVec;
        }
    }

    // ---------------------------------------------------------------
    // Step 4: Write band headers
    // ---------------------------------------------------------------
    // bandBuf[off.band_start + b] = vec4<u32>(count, data_offset, 0, 0)
    // where data_offset = 16 + b * nc (relative offset within element's band region)

    bandBuf[off.band_start + 0u] = vec4<u32>(h0, 16u + 0u * nc, 0u, 0u);
    bandBuf[off.band_start + 1u] = vec4<u32>(h1, 16u + 1u * nc, 0u, 0u);
    bandBuf[off.band_start + 2u] = vec4<u32>(h2, 16u + 2u * nc, 0u, 0u);
    bandBuf[off.band_start + 3u] = vec4<u32>(h3, 16u + 3u * nc, 0u, 0u);
    bandBuf[off.band_start + 4u] = vec4<u32>(h4, 16u + 4u * nc, 0u, 0u);
    bandBuf[off.band_start + 5u] = vec4<u32>(h5, 16u + 5u * nc, 0u, 0u);
    bandBuf[off.band_start + 6u] = vec4<u32>(h6, 16u + 6u * nc, 0u, 0u);
    bandBuf[off.band_start + 7u] = vec4<u32>(h7, 16u + 7u * nc, 0u, 0u);
    bandBuf[off.band_start + BAND_COUNT + 0u] = vec4<u32>(v0, 16u + (BAND_COUNT + 0u) * nc, 0u, 0u);
    bandBuf[off.band_start + BAND_COUNT + 1u] = vec4<u32>(v1, 16u + (BAND_COUNT + 1u) * nc, 0u, 0u);
    bandBuf[off.band_start + BAND_COUNT + 2u] = vec4<u32>(v2, 16u + (BAND_COUNT + 2u) * nc, 0u, 0u);
    bandBuf[off.band_start + BAND_COUNT + 3u] = vec4<u32>(v3, 16u + (BAND_COUNT + 3u) * nc, 0u, 0u);
    bandBuf[off.band_start + BAND_COUNT + 4u] = vec4<u32>(v4, 16u + (BAND_COUNT + 4u) * nc, 0u, 0u);
    bandBuf[off.band_start + BAND_COUNT + 5u] = vec4<u32>(v5, 16u + (BAND_COUNT + 5u) * nc, 0u, 0u);
    bandBuf[off.band_start + BAND_COUNT + 6u] = vec4<u32>(v6, 16u + (BAND_COUNT + 6u) * nc, 0u, 0u);
    bandBuf[off.band_start + BAND_COUNT + 7u] = vec4<u32>(v7, 16u + (BAND_COUNT + 7u) * nc, 0u, 0u);

    // ---------------------------------------------------------------
    // Step 5: Generate vertices + indices
    // ---------------------------------------------------------------
    // 4 vertices per element, 28 floats each (7 x vec4), stride = 112 bytes
    // 6 indices per element

    let scale    = viewport.scale;
    let invScale = viewport.inv_scale;
    let vb_x     = viewport.vb_x;
    let vb_y     = viewport.vb_y;

    // Pixel-space quad corners
    let px0 = (xMin - vb_x) * scale;
    let py0 = (yMin - vb_y) * scale;
    let px1 = (xMax - vb_x) * scale;
    let py1 = (yMax - vb_y) * scale;

    // Pack glyph location: bandBuf offset as flat index
    let glp = bitcast<f32>(off.band_start);

    // Band max packed: ((BAND_COUNT-1) << 16) | (BAND_COUNT-1), with bit 28 for evenodd
    var bandMaxPacked = ((BAND_COUNT - 1u) << 16u) | (BAND_COUNT - 1u);
    if (elem.fill_rule == 1u) { // 1 = evenodd
        bandMaxPacked |= (1u << 28u);
    }
    let bmp = bitcast<f32>(bandMaxPacked);

    // Premultiplied color
    var alpha = elem.fill_a * elem.opacity;
    var cr = elem.fill_r * alpha;
    var cg = elem.fill_g * alpha;
    var cb = elem.fill_b * alpha;

    // Gradient params
    var g0 = vec4<f32>(elem.grad0_x, elem.grad0_y, elem.grad0_z, elem.grad0_w);
    var g1 = vec4<f32>(elem.grad1_x, elem.grad1_y, elem.grad1_z, elem.grad1_w);

    if (elem.gradient_idx >= 0) {
        cr = 0.0; cg = 0.0; cb = 0.0;
        alpha = -elem.opacity;
    }

    // 4 corners unrolled (avoids var array<> bug in simple_wgsl)
    let vertBase = ei * 4u * 28u;

    // Corner 0: px0, py0, normal(-1,-1), em(xMin, yMin)
    {
        let vOff = vertBase + 0u;
        vertexBuf[vOff + 0u] = px0; vertexBuf[vOff + 1u] = py0;
        vertexBuf[vOff + 2u] = -1.0; vertexBuf[vOff + 3u] = -1.0;
        vertexBuf[vOff + 4u] = xMin; vertexBuf[vOff + 5u] = yMin;
        vertexBuf[vOff + 6u] = glp; vertexBuf[vOff + 7u] = bmp;
        vertexBuf[vOff + 8u] = invScale; vertexBuf[vOff + 9u] = 0.0;
        vertexBuf[vOff + 10u] = 0.0; vertexBuf[vOff + 11u] = invScale;
        vertexBuf[vOff + 12u] = bsX; vertexBuf[vOff + 13u] = bsY;
        vertexBuf[vOff + 14u] = boX; vertexBuf[vOff + 15u] = boY;
        vertexBuf[vOff + 16u] = cr; vertexBuf[vOff + 17u] = cg;
        vertexBuf[vOff + 18u] = cb; vertexBuf[vOff + 19u] = alpha;
        vertexBuf[vOff + 20u] = g0.x; vertexBuf[vOff + 21u] = g0.y;
        vertexBuf[vOff + 22u] = g0.z; vertexBuf[vOff + 23u] = g0.w;
        vertexBuf[vOff + 24u] = g1.x; vertexBuf[vOff + 25u] = g1.y;
        vertexBuf[vOff + 26u] = g1.z; vertexBuf[vOff + 27u] = g1.w;
    }

    // Corner 1: px1, py0, normal(1,-1), em(xMax, yMin)
    {
        let vOff = vertBase + 28u;
        vertexBuf[vOff + 0u] = px1; vertexBuf[vOff + 1u] = py0;
        vertexBuf[vOff + 2u] = 1.0; vertexBuf[vOff + 3u] = -1.0;
        vertexBuf[vOff + 4u] = xMax; vertexBuf[vOff + 5u] = yMin;
        vertexBuf[vOff + 6u] = glp; vertexBuf[vOff + 7u] = bmp;
        vertexBuf[vOff + 8u] = invScale; vertexBuf[vOff + 9u] = 0.0;
        vertexBuf[vOff + 10u] = 0.0; vertexBuf[vOff + 11u] = invScale;
        vertexBuf[vOff + 12u] = bsX; vertexBuf[vOff + 13u] = bsY;
        vertexBuf[vOff + 14u] = boX; vertexBuf[vOff + 15u] = boY;
        vertexBuf[vOff + 16u] = cr; vertexBuf[vOff + 17u] = cg;
        vertexBuf[vOff + 18u] = cb; vertexBuf[vOff + 19u] = alpha;
        vertexBuf[vOff + 20u] = g0.x; vertexBuf[vOff + 21u] = g0.y;
        vertexBuf[vOff + 22u] = g0.z; vertexBuf[vOff + 23u] = g0.w;
        vertexBuf[vOff + 24u] = g1.x; vertexBuf[vOff + 25u] = g1.y;
        vertexBuf[vOff + 26u] = g1.z; vertexBuf[vOff + 27u] = g1.w;
    }

    // Corner 2: px1, py1, normal(1,1), em(xMax, yMax)
    {
        let vOff = vertBase + 56u;
        vertexBuf[vOff + 0u] = px1; vertexBuf[vOff + 1u] = py1;
        vertexBuf[vOff + 2u] = 1.0; vertexBuf[vOff + 3u] = 1.0;
        vertexBuf[vOff + 4u] = xMax; vertexBuf[vOff + 5u] = yMax;
        vertexBuf[vOff + 6u] = glp; vertexBuf[vOff + 7u] = bmp;
        vertexBuf[vOff + 8u] = invScale; vertexBuf[vOff + 9u] = 0.0;
        vertexBuf[vOff + 10u] = 0.0; vertexBuf[vOff + 11u] = invScale;
        vertexBuf[vOff + 12u] = bsX; vertexBuf[vOff + 13u] = bsY;
        vertexBuf[vOff + 14u] = boX; vertexBuf[vOff + 15u] = boY;
        vertexBuf[vOff + 16u] = cr; vertexBuf[vOff + 17u] = cg;
        vertexBuf[vOff + 18u] = cb; vertexBuf[vOff + 19u] = alpha;
        vertexBuf[vOff + 20u] = g0.x; vertexBuf[vOff + 21u] = g0.y;
        vertexBuf[vOff + 22u] = g0.z; vertexBuf[vOff + 23u] = g0.w;
        vertexBuf[vOff + 24u] = g1.x; vertexBuf[vOff + 25u] = g1.y;
        vertexBuf[vOff + 26u] = g1.z; vertexBuf[vOff + 27u] = g1.w;
    }

    // Corner 3: px0, py1, normal(-1,1), em(xMin, yMax)
    {
        let vOff = vertBase + 84u;
        vertexBuf[vOff + 0u] = px0; vertexBuf[vOff + 1u] = py1;
        vertexBuf[vOff + 2u] = -1.0; vertexBuf[vOff + 3u] = 1.0;
        vertexBuf[vOff + 4u] = xMin; vertexBuf[vOff + 5u] = yMax;
        vertexBuf[vOff + 6u] = glp; vertexBuf[vOff + 7u] = bmp;
        vertexBuf[vOff + 8u] = invScale; vertexBuf[vOff + 9u] = 0.0;
        vertexBuf[vOff + 10u] = 0.0; vertexBuf[vOff + 11u] = invScale;
        vertexBuf[vOff + 12u] = bsX; vertexBuf[vOff + 13u] = bsY;
        vertexBuf[vOff + 14u] = boX; vertexBuf[vOff + 15u] = boY;
        vertexBuf[vOff + 16u] = cr; vertexBuf[vOff + 17u] = cg;
        vertexBuf[vOff + 18u] = cb; vertexBuf[vOff + 19u] = alpha;
        vertexBuf[vOff + 20u] = g0.x; vertexBuf[vOff + 21u] = g0.y;
        vertexBuf[vOff + 22u] = g0.z; vertexBuf[vOff + 23u] = g0.w;
        vertexBuf[vOff + 24u] = g1.x; vertexBuf[vOff + 25u] = g1.y;
        vertexBuf[vOff + 26u] = g1.z; vertexBuf[vOff + 27u] = g1.w;
    }

    // Indices: two triangles per quad
    let idxBase = ei * 6u;
    let vtxBase = ei * 4u;
    indexBuf[idxBase + 0u] = vtxBase;
    indexBuf[idxBase + 1u] = vtxBase + 1u;
    indexBuf[idxBase + 2u] = vtxBase + 2u;
    indexBuf[idxBase + 3u] = vtxBase;
    indexBuf[idxBase + 4u] = vtxBase + 2u;
    indexBuf[idxBase + 5u] = vtxBase + 3u;
}
