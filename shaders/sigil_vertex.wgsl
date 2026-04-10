// Ultra-minimal vertex shader for debugging simple_wgsl limitations.

struct ParamStruct {
    slug_matrix: array<vec4<f32>, 4>,
    slug_viewport: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: ParamStruct;

struct VertexInput {
    @location(0) pos: vec4<f32>,
    @location(1) tex: vec4<f32>,
    @location(2) jac: vec4<f32>,
    @location(3) bnd: vec4<f32>,
    @location(4) col: vec4<f32>,
    @location(5) grad0: vec4<f32>,
    @location(6) grad1: vec4<f32>,
};

struct VertexStruct {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) texcoord: vec2<f32>,
    @location(2) @interpolate(flat) banding: vec4<f32>,
    @location(3) @interpolate(flat) glyph: vec4<i32>,
    @location(4) @interpolate(flat) grad0: vec4<f32>,
    @location(5) @interpolate(flat) grad1: vec4<f32>,
};

@vertex
fn main(attrib: VertexInput) -> VertexStruct {
    var vresult: VertexStruct;

    // MVP transform: pixel space -> clip space
    let px = attrib.pos.x;
    let py = attrib.pos.y;

    let m0 = params.slug_matrix[0];
    let m1 = params.slug_matrix[1];
    let m2 = params.slug_matrix[2];
    let m3 = params.slug_matrix[3];

    vresult.position = vec4<f32>(
        px * m0.x + py * m0.y + m0.w,
        px * m1.x + py * m1.y + m1.w,
        px * m2.x + py * m2.y + m2.w,
        px * m3.x + py * m3.y + m3.w
    );

    vresult.texcoord = attrib.tex.xy;

    // Unpack glyph data inline (no function call)
    let glyphOffset = bitcast<u32>(attrib.tex.z);
    let flags = bitcast<u32>(attrib.tex.w);
    vresult.glyph = vec4<i32>(
        i32(glyphOffset),
        0,
        i32(flags & 0xFFFFu),
        i32(flags >> 16u)
    );

    vresult.banding = attrib.bnd;
    vresult.color = attrib.col;
    vresult.grad0 = attrib.grad0;
    vresult.grad1 = attrib.grad1;
    return vresult;
}
