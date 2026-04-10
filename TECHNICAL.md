# SigilVG Technical Reference

Complete internals documentation for SigilVG. Covers the full pipeline from SVG parsing through GPU texture packing to the Slug fragment shader.

## Table of contents

1. [Architecture overview](#architecture-overview)
2. [SVG parsing](#svg-parsing)
3. [Curve representation](#curve-representation)
4. [The Slug algorithm](#the-slug-algorithm)
5. [Band spatial partitioning](#band-spatial-partitioning)
6. [GPU data layout](#gpu-data-layout)
7. [Vertex and index buffers](#vertex-and-index-buffers)
8. [Uniform buffer and MVP](#uniform-buffer-and-mvp)
9. [Render pipeline configuration](#render-pipeline-configuration)
10. [Vertex shader](#vertex-shader)
11. [Fragment shader](#fragment-shader)
12. [Gradient rendering](#gradient-rendering)
13. [Text rendering](#text-rendering)
14. [Stroke to fill conversion](#stroke-to-fill-conversion)
15. [Memory management](#memory-management)
16. [Constants and limits](#constants-and-limits)

## Architecture overview

SigilVG follows a four stage pipeline:

```
SVG string
    |
    v
[1. Parse]  sigil_parse_svg()
    |       XML parsing, path commands, transforms, colors, gradients
    |       Output: SigilScene with SigilElements containing quadratic Bezier curves
    v
[2. Prepare]  sigil_prepare()
    |         Band building, curve texture packing, band texture packing,
    |         gradient ramp baking, vertex/index buffer generation, GPU upload
    |         Output: SigilDrawData with GPU buffers and textures
    v
[3. Encode]  sigil_encode()
    |         Records a WebGPU render pass into a command encoder
    |         Binds pipeline, bind group, vertex/index buffers, issues drawIndexed
    v
[4. Submit]  User calls wgpuQueueSubmit()
             GPU executes: vertex shader dilates quads, fragment shader evaluates
             Slug coverage per pixel using curve/band texture lookups
```

All four stages are decoupled. A `SigilScene` can be prepared multiple times at different viewport sizes. `SigilDrawData` can be encoded into multiple command encoders. The uniform buffer can be updated between encodes for camera transforms (the viewer does this for pan/zoom).

## SVG parsing

### XML parser (`sigilvg.h:770`)

A minimal hand written XML parser. Extracts tag names and raw attribute strings without pre-tokenizing attributes. Handles comments (`<!-- -->`), processing instructions (`<?xml ...?>`), and DOCTYPE declarations by skipping them. Self-closing tags (`<rect ... />`) are recognized.

### Attribute resolution (`sigilvg.h:841`)

SVG allows properties in both attributes and inline `style`. SigilVG follows the SVG cascade:

1. `sigil__get_style_prop()` checks the `style` attribute for CSS properties
2. `sigil__get_attr()` checks element attributes
3. `sigil__get_prop()` checks style first, falls back to attributes

### Color parsing (`sigilvg.h:1011`)

Supported formats:

| Format | Example |
|--------|---------|
| 3 digit hex | `#F0C` |
| 6 digit hex | `#FF00CC` |
| RGB function | `rgb(255, 0, 204)` |
| Named colors | 25 SVG colors (black, white, red, blue, orange, etc.) |
| `none` | Transparent (alpha = 0) |
| `url(#id)` | Gradient reference |

All colors are normalized to 0.0 to 1.0 float RGBA.

### Length parsing (`sigilvg.h:935`)

Supports units: `px`, `pt` (96/72 px), `pc` (16 px), `mm`, `cm`, `in`, `em`, `rem`, `%`. Assumes 96 DPI and 16px root font size.

### Transform parsing (`sigilvg.h:1101`)

Parses SVG transform strings into 2x3 affine matrices stored as `float[6]` in row major order: `[a, b, c, d, tx, ty]`.

Supported transforms:

| Transform | Parameters |
|-----------|------------|
| `translate(tx)` or `translate(tx, ty)` | Translation |
| `scale(s)` or `scale(sx, sy)` | Uniform or non uniform scale |
| `rotate(deg)` or `rotate(deg, cx, cy)` | Rotation, optionally around a point |
| `skewX(deg)` | Horizontal skew |
| `skewY(deg)` | Vertical skew |
| `matrix(a,b,c,d,e,f)` | Direct 2x3 matrix |

Multiple transforms in a single attribute are concatenated left to right.

### Transform stack (`sigilvg.h:1902`)

`<g>` elements push transforms onto a stack (max depth: 32). Child elements receive the accumulated group transform multiplied by their own transform. The final transform is applied to all curve control points.

### Path command parsing (`sigilvg.h:491`)

All SVG path commands are supported:

| Command | Meaning | Conversion |
|---------|---------|------------|
| M/m | Move to | Sets subpath start |
| L/l, H/h, V/v | Line to | Stored as degenerate quadratic (midpoint on line) |
| Q/q | Quadratic Bezier | Stored directly |
| T/t | Smooth quadratic | Reflects previous control point |
| C/c | Cubic Bezier | Approximated as quadratics via de Casteljau |
| S/s | Smooth cubic | Reflects previous control point, then cubic to quad |
| A/a | Elliptical arc | Endpoint to center parameterization, split to cubics, then quads |
| Z/z | Close path | Line back to subpath start |

### SVG shape elements

Each shape element is converted to path curves:

| Element | Conversion |
|---------|------------|
| `<rect x y w h>` | 4 line segments forming the rectangle |
| `<circle cx cy r>` | 4 quadratic arcs (90 degree segments) |
| `<ellipse cx cy rx ry>` | 4 quadratic arcs with independent radii |
| `<line x1 y1 x2 y2>` | Single line segment |
| `<polyline points>` | Connected line segments |
| `<polygon points>` | Connected line segments with closure |

### Fill and stroke resolution (`sigilvg.h:2215`)

For each element:

1. Skip if `display:none` or `visibility:hidden`
2. Parse `fill` (default: black) and `fill-opacity`
3. Parse `stroke`, `stroke-width`, `stroke-opacity`
4. If fill references a gradient via `url(#id)`, resolve the gradient index
5. If `fill=none` and stroke is set, perform stroke to fill conversion
6. Multiply `fill-opacity` by element `opacity`
7. Open subpaths are automatically closed for fill (SVG spec)

## Curve representation

Everything in SigilVG is a quadratic Bezier curve:

```c
typedef struct {
    float p0x, p0y;  // Start point
    float p1x, p1y;  // Control point
    float p2x, p2y;  // End point
} SigilCurve;
```

Lines are stored as degenerate quadratics where the control point lies on the line between start and end.

### Cubic to quadratic conversion (`sigilvg.h:356`)

Cubic Bezier curves are approximated by recursive subdivision using de Casteljau splitting at t=0.5.

**Error metric:** The distance between the cubic and quadratic approximation is estimated as `|3*(c1-c2) + p3-p0| / 6`, where c1 and c2 are the cubic control points. If this exceeds 0.25 pixels, the cubic is split in half and each half is processed recursively.

**De Casteljau split at t=0.5:**
```
Given cubic P0, P1, P2, P3:
  m01 = (P0 + P1) / 2
  m12 = (P1 + P2) / 2
  m23 = (P2 + P3) / 2
  m012 = (m01 + m12) / 2
  m123 = (m12 + m23) / 2
  mid = (m012 + m123) / 2

Left half:  P0, m01, m012, mid
Right half: mid, m123, m23, P3
```

Each half cubic is then approximated as a quadratic with control point `(3*(P1 + P2) - P0 - P3) / 4`.

### Arc to curve conversion (`sigilvg.h:386`)

SVG arcs use endpoint parameterization (radii, rotation, large arc flag, sweep flag). Conversion:

1. Convert endpoint parameterization to center parameterization (center, start angle, sweep angle)
2. Correct radii by scaling if the ellipse is too small to reach both endpoints
3. Split the arc into segments of at most pi/2 radians
4. Approximate each segment with a cubic Bezier using tangent length `4/3 * tan(angle/4)`
5. Convert cubics to quadratics via the adaptive method above

## The Slug algorithm

The Slug algorithm (Eric Lengyel, 2017) evaluates vector curve coverage analytically in the fragment shader. The key insight: instead of rasterizing triangles or sampling distance fields, evaluate the winding number contribution of each curve at each pixel.

### Core idea

For each pixel, the shader asks: "how many curves cross the horizontal and vertical rays emanating from this pixel center?" The sum of signed crossings gives the winding number, which determines coverage.

A quadratic Bezier is defined by `B(t) = (1-t)^2 * P0 + 2(1-t)t * P1 + t^2 * P2`. To find horizontal ray crossings, solve `B_y(t) = pixel_y` which is a quadratic equation. Then evaluate `B_x(t)` at each valid root to get the x position of the crossing. If `B_x(t) > pixel_x`, the curve crosses the ray.

### Coverage computation (`sigil_fragment.wgsl:84`)

The fragment shader computes horizontal coverage (`xcov`) and vertical coverage (`ycov`) independently. The final coverage combines them using weights:

```
coverage = max(
    abs(xcov * xwgt + ycov * ywgt) / max(xwgt + ywgt, epsilon),
    min(abs(xcov), abs(ycov))
)
```

The weights (`xwgt`, `ywgt`) measure how close a ray crossing is to the pixel center. A crossing right at the center gets weight 1.0; crossings far from center approach 0.0. This produces smooth antialiased edges.

**Fill rules:**

| Rule | Formula |
|------|---------|
| Nonzero | `saturate(coverage)` |
| Evenodd | `1 - abs(1 - fract(coverage * 0.5) * 2)` |

### Root eligibility (`sigil_fragment.wgsl:23`)

Not every quadratic root contributes to coverage. The `CalcRootCode()` function examines the sign bits of the three control point y coordinates (relative to the pixel) to determine which roots are valid ray crossings. This is encoded as a lookup table packed into the constant `0x2E74` and indexed by 3 sign bits.

### Polynomial solving (`sigil_fragment.wgsl:38`)

`SolveHorizPoly()` solves the quadratic `a*t^2 - 2*b*t + c = 0` where:

```
a = P0 - 2*P1 + P2    (second derivative)
b = P0 - P1            (first derivative at t=0)
c = P0                 (value at t=0)
```

Using the quadratic formula: `t = (b +/- sqrt(b^2 - a*c)) / a`. Degenerate cases (near linear curves where `|a| < 1/65536`) fall back to `t = c / (2*b)`.

The function returns the x positions of the two crossings, already evaluated from the full Bezier polynomial.

## Band spatial partitioning

### Motivation

A naive approach would test every curve against every pixel. For an SVG with 1000 curves, that is 1000 texture lookups and quadratic solves per pixel. Bands reduce this to typically 5 to 20 curves per pixel.

### Structure (`sigilvg.h:1394`)

Each element's bounding box is divided into an 8x8 grid (8 horizontal bands, 8 vertical bands). Each band stores a list of curve indices.

**Band assignment:**

```
bandScale = 8 / (max - min)       // scale factor
bandOffset = -min * bandScale     // offset

For each curve:
  bandMin = clamp(floor(curveMin * bandScale + bandOffset), 0, 7)
  bandMax = clamp(floor(curveMax * bandScale + bandOffset), 0, 7)
  Add curve index to bands [bandMin..bandMax]
```

The same `bandScale` and `bandOffset` are passed to the shader as vertex attributes and used in the fragment shader with identical formulas. This ensures pixel to band mapping is exactly consistent between CPU and GPU.

### Band sorting (`sigilvg.h:2402`)

After assignment, each horizontal band's curves are sorted by descending max x coordinate. Vertical band curves are sorted by descending max y. This enables **early termination** in the shader: when a curve's maximum coordinate is more than 0.5 pixels past the current pixel (in the opposite direction of the ray), all remaining curves in the band are also past, so the loop breaks.

## GPU data layout

### Curve texture

| Property | Value |
|----------|-------|
| Format | RGBA32Float |
| Width | 4096 texels (fixed) |
| Height | `ceil(totalCurves * 2 / 4096)` |

Each quadratic Bezier occupies 2 consecutive texels:

```
Texel 0: (p0.x, p0.y, p1.x, p1.y)
Texel 1: (p2.x, p2.y, unused, unused)
```

Curves are packed row major. The (x, y) texel coordinates of each curve are stored in the band texture for lookup.

### Band texture

| Property | Value |
|----------|-------|
| Format | RGBA32Uint |
| Width | 4096 texels (fixed) |
| Height | Computed from total band data size |

Layout per element:

```
[Header: 16 texels] [hBand 0 data] ... [hBand 7 data] [vBand 0 data] ... [vBand 7 data]
```

**Header texels (one per band, 16 total):**

Each header texel contains `(count, offset, 0, 0)` where `count` is the number of curves in the band and `offset` is the byte offset from the element's start to the band's curve index list.

**Data texels (one per curve index):**

Each data texel contains `(curve_tex_x, curve_tex_y, 0, 0)` pointing to the curve's location in the curve texture.

**Row wrapping:** if the header would straddle a row boundary in the 4096 wide texture, padding is inserted to keep the header contiguous.

### Gradient ramp texture

| Property | Value |
|----------|-------|
| Format | RGBA8Unorm |
| Width | 256 texels |
| Height | Number of gradients |

Each row is a 256 sample gradient ramp, linearly interpolated between stops. Sampled in the fragment shader with a linear sampler at `(t, row + 0.5/height)`.

## Vertex and index buffers

### Vertex format

Each element becomes a quad (4 vertices, 6 indices). Each vertex has 7 vec4 attributes (112 bytes per vertex):

| Attribute | Location | Contents |
|-----------|----------|----------|
| `pos` | 0 | `(pixel_x, pixel_y, normal_x, normal_y)` |
| `tex` | 1 | `(em_x, em_y, glyph_loc_packed, band_maxes_packed)` |
| `jac` | 2 | `(j00, j01, j10, j11)` inverse Jacobian |
| `bnd` | 3 | `(bandScale_x, bandScale_y, bandOffset_x, bandOffset_y)` |
| `col` | 4 | `(R, G, B, A)` premultiplied; negative A = gradient mode |
| `grad0` | 5 | Linear: `(x1, y1, x2, y2)`, Radial: `(cx, cy, fx, fy)` |
| `grad1` | 6 | `(gradTexRow, gradType, radius, spread)` |

### Packed fields

**`tex.z` (glyph location):**
```
bits 0..15:   x coordinate in band texture
bits 16..31:  y coordinate in band texture
```
Interpreted via `bitcast<u32>` in the vertex shader.

**`tex.w` (band maxes and flags):**
```
bits 0..7:    max horizontal band index
bits 16..23:  max vertical band index
bit 28:       evenodd fill rule flag
```

### Quad geometry

The four vertices are the corners of the element's bounding box in pixel space. The normal vectors (`pos.zw`) point outward from the center and are used by the Slug dilation algorithm to expand the quad slightly for proper edge coverage.

### Index buffer

UINT32 format. 6 indices per element: `[base+0, base+1, base+2, base+0, base+2, base+3]`.

## Uniform buffer and MVP

The uniform buffer is 80 bytes (20 floats):

```
Floats 0..15:   4x4 MVP matrix (row major)
Floats 16..17:  viewport width, viewport height
Floats 18..19:  padding (zero)
```

### MVP construction (`sigilvg.h:2912`)

The default MVP converts pixel coordinates to clip space with Y flip:

```
| 2/vw    0      0    -1  |      x_clip = 2*x/vw - 1
|  0    -2/vh    0     1  |      y_clip = -2*y/vh + 1
|  0      0      0     0  |      z_clip = 0
|  0      0      0     1  |      w_clip = 1
```

The viewer overrides this with a camera transform that includes zoom and pan:

```
| 2z/vw     0      0    -cam_x*2z/vw |
|  0     -2z/vh    0     cam_y*2z/vh  |
|  0        0      0         0        |
|  0        0      0         1        |
```

Where `z` is the zoom factor and `(cam_x, cam_y)` is the camera center in pixel space. The UBO can be rewritten via `wgpuQueueWriteBuffer` each frame without rebuilding draw data.

## Render pipeline configuration

### Shader loading (`sigilvg.h:2469`)

Shaders are loaded from disk at `sigil_create` time:

1. Check `SIGIL_SHADER_PATH` environment variable
2. Fall back to current working directory
3. Load `sigil_vertex.wgsl` and `sigil_fragment.wgsl`
4. Create `WGPUShaderModule` from WGSL source

### Bind group layout (`sigilvg.h:2506`)

| Binding | Stage | Type | Format |
|---------|-------|------|--------|
| 0 | Vertex | Uniform buffer | 80 bytes |
| 1 | Fragment | Texture 2D | RGBA32Float (curves) |
| 2 | Fragment | Texture 2D | RGBA32Uint (bands) |
| 3 | Fragment | Texture 2D | RGBA32Float (gradient ramp) |
| 4 | Fragment | Sampler | Linear filtering |

### Pipeline state

| Setting | Value |
|---------|-------|
| Topology | Triangle list |
| Cull mode | None |
| Front face | CCW |
| Color blend | Premultiplied alpha: `src=One, dst=OneMinusSrcAlpha, op=Add` |
| Alpha blend | Same as color |
| Depth stencil | Optional, LessEqual, write enabled |
| Vertex stride | 112 bytes (7 x vec4f) |

## Vertex shader

File: `shaders/sigil_vertex.wgsl`

### SlugDilate (`sigil_vertex.wgsl:66`)

The Slug algorithm requires the quad to extend slightly beyond the element's bounding box so that antialiased edge pixels are rendered. `SlugDilate` computes a per vertex dilation that depends on the MVP matrix and the vertex normal.

**Inputs:** vertex position, normal, em space coordinate, inverse Jacobian, and three rows of the MVP matrix.

**Algorithm:**

```
n = normalize(normal)
s = dot(M3.xy, pos.xy) + M3.w        // clip-space w denominator
t = dot(M3.xy, n)                     // rate of change of w along normal

u = (s * dot(M0.xy, n) - t * (dot(M0.xy, pos.xy) + M0.w)) * viewport_w
v = (s * dot(M1.xy, n) - t * (dot(M1.xy, pos.xy) + M1.w)) * viewport_h

d = normal * (s^2 * (s*t + sqrt(u^2 + v^2)) / (u^2 + v^2 - s^2*t^2))

dilated_pos = pos + d
dilated_texcoord = tex + J^-1 * d     // J^-1 is the inverse Jacobian
```

This produces exactly enough dilation for the fragment shader to have valid coverage at all edge pixels.

### SlugUnpack (`sigil_vertex.wgsl:49`)

Extracts the packed integer fields from `tex.zw`:

```
g = bitcast<u32>(tex.zw)
glyph = ivec4(g.x & 0xFFFF, g.x >> 16, g.y & 0xFFFF, g.y >> 16)
```

The glyph data vector contains `(bandTexX, bandTexY, maxBandX, maxBandYAndFlags)`.

## Fragment shader

File: `shaders/sigil_fragment.wgsl`

### SlugRender (`sigil_fragment.wgsl:104`)

The main coverage computation function. Receives the curve texture, band texture, em space pixel coordinate, band transform, and unpacked glyph data.

**Step 1: Band lookup**

```
emsPerPixel = fwidth(renderCoord)
pixelsPerEm = 1.0 / emsPerPixel
bandIndex = clamp(renderCoord * bandScale + bandOffset, 0, bandMax)
```

**Step 2: Horizontal coverage**

Read the horizontal band header from the band texture. For each curve in the band:

1. Load the curve's 3 control points from the curve texture
2. Subtract `renderCoord` to make them pixel relative
3. Early exit if all x values are past the pixel (sorted band optimization)
4. Compute root eligibility via `CalcRootCode(y1, y2, y3)`
5. If roots exist, solve `SolveHorizPoly` for x intercepts
6. Accumulate signed coverage: `xcov += saturate(x + 0.5)` for valid crossings

**Step 3: Vertical coverage**

Same as horizontal but with x and y swapped. Vertical bands are stored after horizontal bands in the band texture header (offset by `bandMax.y + 1`).

**Step 4: Combine**

```
coverage = CalcCoverage(xcov, ycov, xwgt, ywgt, flags)
```

Returns a float in [0, 1] representing the pixel's vector coverage.

## Gradient rendering

### Gradient parsing (`sigilvg.h:133`)

Gradients are parsed from `<linearGradient>` and `<radialGradient>` elements:

```c
typedef struct {
    char id[128];
    int type;             // 1 = linear, 2 = radial
    SigilGradientStop *stops;
    int stop_count;
    float x1, y1, x2, y2; // Linear endpoints
    float cx, cy, r;       // Radial center, radius
    float fx, fy, fr;      // Radial focal point, focal radius
    float transform[6];    // gradientTransform
    int objectBBox;        // 1 = objectBoundingBox, 0 = userSpaceOnUse
    int spread;            // 0 = pad, 1 = reflect, 2 = repeat
    char href[128];        // xlink:href for stop inheritance
} SigilGradientDef;
```

### `xlink:href` inheritance (`sigilvg.h:179`)

When a gradient references another via `xlink:href`, it inherits the target's stops. Resolved during `sigil_prepare`.

### Gradient ramp baking (`sigilvg.h:201`)

Each gradient becomes a 256 pixel row in the gradient ramp texture. For each of the 256 positions (t = 0.0 to 1.0):

1. Find the two stops bracketing t
2. Linearly interpolate the color
3. Clamp to the first/last stop color outside the stop range

### Fragment shader gradient evaluation (`sigil_fragment.wgsl:217`)

Gradient mode is signaled by negative alpha in the vertex color.

**Linear gradient:** project the pixel's em space coordinate onto the gradient line to get parametric t:
```
d = p2 - p1
t = dot(pixel - p1, d) / dot(d, d)
```

**Radial gradient:** compute distance from center normalized by radius:
```
t = length(pixel - center) / radius
```

**Spread modes:** applied to t before sampling:

| Mode | Formula |
|------|---------|
| Pad | `clamp(t, 0, 1)` |
| Reflect | Fold t into [0, 1] by reflecting at boundaries |
| Repeat | `fract(t)` |

The ramp texture is sampled at `(t, gradRow)` with linear filtering.

## Text rendering

### Font loading (`sigilvg.h:1652`)

`sigil_load_font()` parses TTF data via stb_truetype and associates it with a family name. Multiple fonts can be loaded; `<text>` elements match by `font-family` attribute with fallback to the first loaded font.

### Glyph extraction (`sigilvg.h:1675`)

For each glyph:

1. `stbtt_GetGlyphShape()` returns move/line/quadratic/cubic vertices
2. Lines become degenerate quadratics
3. Cubic vertices are split to quadratics via de Casteljau
4. All curves are scaled and positioned by font size and cursor position

### Text layout (`sigilvg.h:1757`)

1. UTF 8 decode each codepoint
2. Look up glyph index via `stbtt_FindGlyphIndex()`
3. Get advance width via `stbtt_GetGlyphHMetrics()`
4. Apply kerning via `stbtt_GetGlyphKernAdvance()`
5. Extract glyph curves, apply transform: `x' = cursor + glyph_x * scale`, `y' = baseline - glyph_y * scale` (Y flip for SVG coordinates)
6. Advance cursor by `advance * scale`

## Stroke to fill conversion

When an element has `fill=none` and a stroke, the stroke is converted to a filled outline (`sigilvg.h:1475`):

### Algorithm

1. **Flatten curves to polyline** (`sigilvg.h:1446`): adaptive subdivision with 0.5px tolerance
2. **Compute normals**: at each vertex, average the normals of the two adjacent segments
3. **Generate outline**: for each vertex, emit two points offset by `+/- stroke_width/2` along the averaged normal
4. **Emit curves**: connect left side points as quadratic curves (forward), then right side points (backward)
5. **Close contour**: connect the last right point back to the first left point

The result is a filled closed contour that approximates the stroked path with rounded joins.

## Memory management

### Ownership model

```
SigilScene
    owns: elements[], gradients[], fonts[], font_names[]
    each element owns: curves[], bandData (hBands/vBands curve index lists)

SigilDrawData
    owns: vertexBuffer, indexBuffer, uniformBuffer (WGPUBuffer)
          curveTexture, bandTexture, gradientTexture (WGPUTexture)
          curveView, bandView, gradientView (WGPUTextureView)
          gradientSampler (WGPUSampler)
          bindGroup (WGPUBindGroup)

SigilContext
    owns: pipeline, bindGroupLayout, pipelineLayout (WGPURenderPipeline, etc.)
          vertexShader, fragmentShader (WGPUShaderModule)
          device reference (not owned)
```

### Cleanup order

No strict ordering is required between the three objects, but all GPU resources in `SigilDrawData` and `SigilContext` must be released before the `WGPUDevice` is destroyed.

### Temporary allocations

Path strings, transform strings, and color strings are malloc'd during parsing and freed immediately. Staging data for texture uploads is written via `wgpuQueueWriteBuffer/Texture` (no explicit free needed; the driver manages the copy).

## Constants and limits

| Constant | Value | Purpose |
|----------|-------|---------|
| `SIGIL_TEX_WIDTH` | 4096 | Fixed width for curve and band textures |
| `SIGIL_BAND_COUNT` | 8 | Number of bands per axis (8x8 grid) |
| `SIGIL_GRADIENT_RAMP_WIDTH` | 256 | Gradient ramp texture width |
| Transform stack depth | 32 | Maximum `<g>` nesting depth |
| `kLogBandTextureWidth` | 12 | log2(4096), used in shader for address wrapping |
| Cubic to quad tolerance | 0.25 | Maximum pixel error for cubic approximation |
| Polyline flatten tolerance | 0.5 | Maximum pixel error for curve flattening |
| Near linear threshold | 1/65536 | Below this, quadratic degrades to linear solve |
