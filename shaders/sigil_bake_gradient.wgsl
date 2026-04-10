// ===================================================
// GPU-driven gradient ramp baking compute shader for SigilVG
// Replaces the CPU sigil__bake_gradient_ramp() path.
// Dispatched as dispatch(gradient_count, 1, 1), workgroup size 256.
// Each thread bakes one texel of a 256-wide gradient ramp row.
// ===================================================

struct GradientDef {
    type_       : i32,
    spread      : i32,
    stop_count  : i32,
    stop_offset : i32,
    objectBBox  : i32,
    pad0        : i32,
    pad1        : i32,
    pad2        : i32,
    x1 : f32, y1 : f32, x2 : f32, y2 : f32,
    cx : f32, cy : f32, r  : f32,
    fx : f32, fy : f32, fr : f32,
    t0 : f32, t1 : f32, t2 : f32, t3 : f32, t4 : f32, t5 : f32,
};

struct GradientStop {
    color  : vec4<f32>,
    offset : f32,
    pad0   : f32,
    pad1   : f32,
    pad2   : f32,
};

@group(0) @binding(0) var<storage, read> gradients : array<GradientDef>;
@group(0) @binding(1) var<storage, read> stops     : array<GradientStop>;
@group(0) @binding(2) var<storage, read_write> rampBuf : array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let gradIdx = gid.x / 256u;
    let texelIdx = gid.x % 256u;

    let grad = gradients[gradIdx];
    let sc = u32(grad.stop_count);
    let so = u32(grad.stop_offset);

    var color = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    if (sc == 1u) {
        color = stops[so].color;
    } else if (sc >= 2u) {
        let t = f32(texelIdx) / 255.0;

        // Linear search for bracketing stops.
        // No break (simple_wgsl codegen issue), last match wins.
        var lo = 0u;
        var hi = 1u;
        for (var s = 0u; s < 31u; s++) {
            if (s < sc - 1u) {
                let sOff  = stops[so + s].offset;
                let sOff1 = stops[so + s + 1u].offset;
                if (t >= sOff && t <= sOff1) {
                    lo = s;
                    hi = s + 1u;
                }
            }
        }

        // Clamp to edge stops
        if (t <= stops[so].offset) {
            lo = 0u; hi = 0u;
        }
        if (sc >= 2u && t >= stops[so + sc - 1u].offset) {
            lo = sc - 1u; hi = sc - 1u;
        }

        let loC = stops[so + lo].color;
        let hiC = stops[so + hi].color;
        let seg = stops[so + hi].offset - stops[so + lo].offset;
        var u_val = select(0.0, (t - stops[so + lo].offset) / seg, seg > 1e-6);
        u_val = clamp(u_val, 0.0, 1.0);
        color = mix(loC, hiC, u_val);
    }

    color = clamp(color, vec4<f32>(0.0), vec4<f32>(1.0));
    rampBuf[gradIdx * 256u + texelIdx] = pack4x8unorm(color);
}
