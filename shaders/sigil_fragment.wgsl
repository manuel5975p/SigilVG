// ===================================================
// Reference pixel shader for the Slug algorithm ported to WGSL
// This code is made available under the MIT License.
// Copyright 2017, by Eric Lengyel.
// ===================================================


// The curve and band textures use a fixed width of 4096 texels.

const kLogBandTextureWidth: u32 = 12u;

// It's convenient to have a texel load function to aid in translation to other shader languages.

fn TexelLoad2D_f32(tex: texture_2d<f32>, coords: vec2<i32>) -> vec4<f32> {
    return textureLoad(tex, coords, 0);
}

fn TexelLoad2D_u32(tex: texture_2d<u32>, coords: vec2<i32>) -> vec4<u32> {
    return textureLoad(tex, coords, 0);
}


fn CalcRootCode(y1: f32, y2: f32, y3: f32) -> u32 {
    // Calculate the root eligibility code for a sample-relative quadratic Bezier curve.
    // Extract the signs of the y coordinates of the three control points.

    let i1 = bitcast<u32>(y1) >> 31u;
    let i2 = bitcast<u32>(y2) >> 30u;
    let i3 = bitcast<u32>(y3) >> 29u;

    let shift = (i3 & 4u) | (((i2 & 2u) | (i1 & ~2u)) & ~4u);

    // Eligibility is returned in bits 0 and 8.

    return ((0x2E74u >> shift) & 0x0101u);
}

fn SolveHorizPoly(p12: vec4<f32>, p3: vec2<f32>) -> vec2<f32> {
    let a = vec2<f32>(p12.x - p12.z * 2.0 + p3.x, p12.y - p12.w * 2.0 + p3.y);
    let b = vec2<f32>(p12.x - p12.z, p12.y - p12.w);
    let ra = 1.0 / a.y;
    let rb = 0.5 / b.y;

    let d = sqrt(max(b.y * b.y - a.y * p12.y, 0.0));
    var t1 = (b.y - d) * ra;
    var t2 = (b.y + d) * ra;

    if (abs(a.y) < 1.0 / 65536.0) {
        t1 = p12.y * rb;
        t2 = p12.y * rb;
    }

    return vec2<f32>((a.x * t1 - b.x * 2.0) * t1 + p12.x, (a.x * t2 - b.x * 2.0) * t2 + p12.x);
}

fn SolveVertPoly(p12: vec4<f32>, p3: vec2<f32>) -> vec2<f32> {
    let a = vec2<f32>(p12.x - p12.z * 2.0 + p3.x, p12.y - p12.w * 2.0 + p3.y);
    let b = vec2<f32>(p12.x - p12.z, p12.y - p12.w);
    let ra = 1.0 / a.x;
    let rb = 0.5 / b.x;

    let d = sqrt(max(b.x * b.x - a.x * p12.x, 0.0));
    var t1 = (b.x - d) * ra;
    var t2 = (b.x + d) * ra;

    if (abs(a.x) < 1.0 / 65536.0) {
        t1 = p12.x * rb;
        t2 = p12.x * rb;
    }

    return vec2<f32>((a.y * t1 - b.y * 2.0) * t1 + p12.y, (a.y * t2 - b.y * 2.0) * t2 + p12.y);
}

fn CalcBandLoc(glyphLoc: vec2<i32>, offset: u32) -> vec2<i32> {
    var bandLoc = vec2<i32>(glyphLoc.x + i32(offset), glyphLoc.y);
    bandLoc.y += bandLoc.x >> kLogBandTextureWidth;
    bandLoc.x &= (1 << kLogBandTextureWidth) - 1;
    return bandLoc;
}

override SLUG_EVENODD: bool = true;
override SLUG_WEIGHT: bool = false;

fn CalcCoverage(xcov: f32, ycov: f32, xwgt: f32, ywgt: f32, flags: i32) -> f32 {
    var coverage = max(abs(xcov * xwgt + ycov * ywgt) / max(xwgt + ywgt, 1.0 / 65536.0), min(abs(xcov), abs(ycov)));

    if (SLUG_EVENODD) {
        if ((flags & 0x1000) == 0) {
            coverage = saturate(coverage);
        } else {
            coverage = 1.0 - abs(1.0 - fract(coverage * 0.5) * 2.0);
        }
    } else {
        coverage = saturate(coverage);
    }

    if (SLUG_WEIGHT) {
        coverage = sqrt(coverage);
    }

    return coverage;
}

fn SlugRender(curveData: texture_2d<f32>, bandData: texture_2d<u32>, renderCoord: vec2<f32>, bandTransform: vec4<f32>, glyphData: vec4<i32>) -> f32 {
    var curveIndex: i32;

    let emsPerPixel = fwidth(renderCoord);
    let pixelsPerEm = 1.0 / emsPerPixel;

    var bandMax = vec2<i32>(glyphData.z, glyphData.w & 0x00FF);

    let bandIndex = clamp(vec2<i32>(renderCoord * bandTransform.xy + bandTransform.zw), vec2<i32>(0, 0), bandMax);
    let glyphLoc = vec2<i32>(glyphData.x, glyphData.y);

    var xcov: f32 = 0.0;
    var xwgt: f32 = 0.0;

    let hbandData = TexelLoad2D_u32(bandData, vec2<i32>(glyphLoc.x + bandIndex.y, glyphLoc.y)).xy;
    let hbandLoc = CalcBandLoc(glyphLoc, hbandData.y);

    for (curveIndex = 0; curveIndex < i32(hbandData.x); curveIndex++) {
        let curveLoc = vec2<i32>(TexelLoad2D_u32(bandData, vec2<i32>(hbandLoc.x + curveIndex, hbandLoc.y)).xy);
        let p12 = TexelLoad2D_f32(curveData, curveLoc) - vec4<f32>(renderCoord, renderCoord);
        let p3 = TexelLoad2D_f32(curveData, vec2<i32>(curveLoc.x + 1, curveLoc.y)).xy - renderCoord;

        if (max(max(p12.x, p12.z), p3.x) * pixelsPerEm.x < -0.5) {
            break;
        }

        let code = CalcRootCode(p12.y, p12.w, p3.y);
        if (code != 0u) {
            let r = SolveHorizPoly(p12, p3) * pixelsPerEm.x;

            if ((code & 1u) != 0u) {
                xcov += saturate(r.x + 0.5);
                xwgt = max(xwgt, saturate(1.0 - abs(r.x) * 2.0));
            }

            if (code > 1u) {
                xcov -= saturate(r.y + 0.5);
                xwgt = max(xwgt, saturate(1.0 - abs(r.y) * 2.0));
            }
        }
    }

    var ycov: f32 = 0.0;
    var ywgt: f32 = 0.0;

    let vbandData = TexelLoad2D_u32(bandData, vec2<i32>(glyphLoc.x + bandMax.y + 1 + bandIndex.x, glyphLoc.y)).xy;
    let vbandLoc = CalcBandLoc(glyphLoc, vbandData.y);

    for (curveIndex = 0; curveIndex < i32(vbandData.x); curveIndex++) {
        let curveLoc = vec2<i32>(TexelLoad2D_u32(bandData, vec2<i32>(vbandLoc.x + curveIndex, vbandLoc.y)).xy);
        let p12 = TexelLoad2D_f32(curveData, curveLoc) - vec4<f32>(renderCoord, renderCoord);
        let p3 = TexelLoad2D_f32(curveData, vec2<i32>(curveLoc.x + 1, curveLoc.y)).xy - renderCoord;

        if (max(max(p12.y, p12.w), p3.y) * pixelsPerEm.y < -0.5) {
            break;
        }

        let code = CalcRootCode(p12.x, p12.z, p3.x);
        if (code != 0u) {
            let r = SolveVertPoly(p12, p3) * pixelsPerEm.y;

            if ((code & 1u) != 0u) {
                ycov -= saturate(r.x + 0.5);
                ywgt = max(ywgt, saturate(1.0 - abs(r.x) * 2.0));
            }

            if (code > 1u) {
                ycov += saturate(r.y + 0.5);
                ywgt = max(ywgt, saturate(1.0 - abs(r.y) * 2.0));
            }
        }
    }

    return CalcCoverage(xcov, ycov, xwgt, ywgt, glyphData.w);
}

// Apply spread method to parametric t value
fn applySpread(t_in: f32, spread: i32) -> f32 {
    var t = t_in;
    if (spread == 1) {
        // reflect
        t = abs(t);
        let period = t - floor(t / 2.0) * 2.0;
        if (period > 1.0) { t = 2.0 - period; } else { t = period; }
    } else if (spread == 2) {
        // repeat
        t = t - floor(t);
    } else {
        // pad (default)
        t = clamp(t, 0.0, 1.0);
    }
    return t;
}

struct VertexStruct {
    @builtin(position) position: vec4<f32>,              // Clip-space vertex position.
    @location(0) color: vec4<f32>,                       // Vertex color.
    @location(1) texcoord: vec2<f32>,                    // Em-space sample coordinates.
    @location(2) @interpolate(flat) banding: vec4<f32>,  // Band scale and offset, constant over glyph.
    @location(3) @interpolate(flat) glyph: vec4<i32>,    // Glyph data.
    @location(4) @interpolate(flat) grad0: vec4<f32>,    // Gradient params 0.
    @location(5) @interpolate(flat) grad1: vec4<f32>,    // Gradient params 1.
};

@group(0) @binding(1) var curveTexture: texture_2d<f32>;     // Control point texture.
@group(0) @binding(2) var bandTexture: texture_2d<u32>;      // Band data texture.
@group(0) @binding(3) var gradientTexture: texture_2d<f32>;  // Gradient ramp texture.
@group(0) @binding(4) var gradientSampler: sampler;          // Linear sampler for gradients.

@fragment
fn main(vresult: VertexStruct) -> @location(0) vec4<f32> {
    let coverage = SlugRender(curveTexture, bandTexture, vresult.texcoord, vresult.banding, vresult.glyph);

    // Check if this is a gradient fill (negative alpha signals gradient mode)
    if (vresult.color.a < 0.0) {
        let opacity = -vresult.color.a;
        let gradType = i32(vresult.grad1.y);
        let gradRow = vresult.grad1.x;
        let spread = i32(vresult.grad1.w);

        var t: f32 = 0.0;

        if (gradType == 1) {
            // Linear gradient: project renderCoord onto gradient line
            let p1 = vresult.grad0.xy;
            let p2 = vresult.grad0.zw;
            let d = p2 - p1;
            let len2 = dot(d, d);
            if (len2 > 0.0) {
                t = dot(vresult.texcoord - p1, d) / len2;
            }
        } else {
            // Radial gradient
            let center = vresult.grad0.xy;
            let focal = vresult.grad0.zw;
            let radius = vresult.grad1.z;
            if (radius > 0.0) {
                let dist = length(vresult.texcoord - center);
                t = dist / radius;
            }
        }

        t = applySpread(t, spread);

        // Sample gradient ramp texture
        let gradColor = textureSampleLevel(gradientTexture, gradientSampler,
            vec2<f32>(t, gradRow), 0.0);

        // Premultiply alpha and apply coverage + element opacity
        let a = gradColor.a * opacity * coverage;
        return vec4<f32>(gradColor.rgb * opacity * coverage, a);
    }

    // Solid fill: premultiplied color * coverage
    return vresult.color * coverage;
}
