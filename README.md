# SigilVG

GPU accelerated SVG rendering in a single C header.

SigilVG turns SVG paths into resolution independent, antialiased vector graphics on the GPU using the **Slug algorithm**: quadratic Bezier curves evaluated analytically in the fragment shader. No tessellation. No texture atlases. Just math on pixels.

Built on [WGVK](https://github.com/manuel5975p/WGVK) (WebGPU over Vulkan). Runs on Linux, macOS, Windows, and the web via Emscripten.

> For a deep dive into the rendering pipeline, data layout, and shader internals, see **[TECHNICAL.md](TECHNICAL.md)**.

## What it renders

| Feature | Status |
|---------|--------|
| `<path>` (all commands: M L H V C S Q T A Z) | Full |
| `<rect>`, `<circle>`, `<ellipse>` | Full |
| `<line>`, `<polyline>`, `<polygon>` | Full |
| `<text>` via stb_truetype | Full |
| `<linearGradient>`, `<radialGradient>` | Full (with stops, spread, href inheritance) |
| `<g>` with nested transforms | Full (up to 32 levels) |
| Fill rules (nonzero, evenodd) | Full |
| Stroke to fill conversion | Full |
| Opacity, inline CSS styles | Full |

## Quick start

### 1. Build

```sh
cmake -B build -G Ninja
ninja -C build
```

This fetches all dependencies automatically (WGVK, GLFW, stb headers, PlutoSVG for tests).

### 2. Run

```sh
# Headless render to PNG
./build/sigilvg_headless my_drawing.svg output.png

# Interactive viewer with pan/zoom
./build/sigilvg_viewer my_drawing.svg
```

**Viewer controls:** scroll to zoom, middle/right drag to pan, R to reset, Escape to quit.

### 3. Try the demos

```sh
./build/sigilvg_headless demos/gradient_scene.svg sunset.png
./build/sigilvg_headless demos/shapes.svg shapes.png
./build/sigilvg_viewer demos/shapes.svg
```

## Using the library

SigilVG is a single header library. Define `SIGIL_IMPLEMENTATION` in exactly one `.c` file before including it.

```c
#define SIGIL_IMPLEMENTATION
#include "sigilvg.h"
```

### Minimal example

```c
// 1. Create the GPU pipeline (once)
SigilContext *ctx = sigil_create(device, WGPUTextureFormat_RGBA8Unorm, 0);

// 2. Parse an SVG string
SigilScene *scene = sigil_parse_svg(svg_data, svg_len);

// 3. Prepare draw data (builds curve textures, band structures, vertex buffers)
SigilDrawData *dd = sigil_prepare(ctx, scene, 800.0f, 600.0f, false);

// 4. Encode the render pass
WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device, NULL);
float bg[] = {1, 1, 1, 1};
sigil_encode(ctx, dd, enc, colorView, NULL, bg);

// 5. Submit
WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
wgpuQueueSubmit(queue, 1, &cb);

// 6. Cleanup
sigil_free_draw_data(dd);
sigil_free_scene(scene);
sigil_destroy(ctx);
```

### Text rendering

```c
SigilScene *scene = sigil_parse_svg(svg_with_text, len);

// Load a TTF font and bind it to a family name
unsigned char *font = load_file("Inter.ttf", &font_size);
sigil_load_font(scene, "Inter", font, font_size);

// Font data must stay alive for the lifetime of the scene
SigilDrawData *dd = sigil_prepare(ctx, scene, 800, 600, false);
```

## API reference

```c
// Lifecycle
SigilContext*  sigil_create(WGPUDevice device, WGPUTextureFormat colorFmt, WGPUTextureFormat depthFmt);
void           sigil_destroy(SigilContext *ctx);

// Scene
SigilScene*    sigil_parse_svg(const char *svg_data, size_t len);
int            sigil_load_font(SigilScene *scene, const char *family, const unsigned char *ttf, size_t size);
void           sigil_free_scene(SigilScene *scene);

// Rendering
SigilDrawData* sigil_prepare(SigilContext *ctx, SigilScene *scene, float w, float h, bool depth);
void           sigil_encode(SigilContext *ctx, SigilDrawData *dd, WGPUCommandEncoder enc,
                            WGPUTextureView color, WGPUTextureView depth, const float clear[4]);
void           sigil_free_draw_data(SigilDrawData *dd);
```

**Depth buffering:** pass a non zero `depthFormat` to `sigil_create` and `depth=true` to `sigil_prepare` for correct z ordering in mixed 3D/2D scenes.

**Clear color:** pass `NULL` to `sigil_encode` to use `LoadOp_Load` (compositing on top of existing content) instead of clearing.

## Build targets

| Target | Description |
|--------|-------------|
| `sigilvg_headless` | Offscreen SVG to PNG renderer |
| `sigilvg_viewer` | Interactive GLFW windowed viewer |
| `sigilvg_test_parse` | SVG parsing unit tests |
| `sigilvg_test_ref` | Pixel comparison vs PlutoSVG reference |
| `sigilvg_headless_batch` | Batch render multiple SVGs |
| `sigilvg_batch_pixel` | Batch pixel comparison tests |

### Web (Emscripten)

```sh
emcmake cmake -B build_web -G Ninja
ninja -C build_web
```

Produces `sigilvg_web_demo.html` with drag and drop SVG loading. Uses `emdawnwebgpu` for browser WebGPU.

## Dependencies

All fetched automatically by CMake:

| Dependency | Purpose |
|------------|---------|
| [WGVK](https://github.com/manuel5975p/WGVK) | WebGPU implementation on Vulkan 1.1 |
| [GLFW 3.4](https://www.glfw.org/) | Windowing (viewer only) |
| [stb_truetype](https://github.com/nothings/stb) | TTF font parsing |
| [stb_image_write](https://github.com/nothings/stb) | PNG output |
| [PlutoSVG](https://github.com/nickthorpe/plutosvg) | Reference renderer (tests only) |

## Project layout

```
sigilvg.h                      The entire library
shaders/sigil_vertex.wgsl      Vertex shader (Slug dilation + MVP)
shaders/sigil_fragment.wgsl    Fragment shader (Slug coverage + gradients)
examples/headless.c            Offscreen render to PNG
examples/viewer.c              GLFW interactive viewer
examples/web_demo.c            Emscripten WebGPU demo
tests/test_parse.c             Parsing unit tests
tests/test_reference.c         Pixel regression tests
demos/                         Sample SVG files
```

## License

Slug algorithm shaders: MIT License, Copyright 2017 Eric Lengyel.
