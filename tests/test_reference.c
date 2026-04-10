/*
 * test_reference.c — SigilVG vs PlutoSVG comparison tests.
 *
 * Renders each SVG via both:
 *   1. PlutoSVG (CPU reference rasterizer)
 *   2. SigilVG  (GPU via WGVK headless)
 *
 * Then compares the two RGBA images with PSNR and differing-pixel counts.
 * Writes diff images for visual inspection.
 */

#define SIGIL_IMPLEMENTATION
#include "sigilvg.h"

#include <plutovg.h>
#include <plutosvg.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/*  Test harness                                                      */
/* ------------------------------------------------------------------ */

static int g_pass = 0, g_fail = 0;

#define TEST(name) static void name(void)
#define ASSERT(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        g_fail++; return; \
    } \
} while(0)
#define RUN(name) do { \
    int _prev_fail = g_fail; \
    printf("  %-40s", #name "..."); fflush(stdout); name(); \
    if (g_fail == _prev_fail) { printf(" ok\n"); g_pass++; } \
    else { printf(" FAILED\n"); } \
} while(0)

/* ------------------------------------------------------------------ */
/*  Image type                                                        */
/* ------------------------------------------------------------------ */

typedef struct {
    unsigned char *pixels; /* RGBA, row-major */
    int w, h;
} Image;

static void image_free(Image *img) { free(img->pixels); img->pixels = NULL; }

/* ------------------------------------------------------------------ */
/*  PlutoSVG reference renderer                                       */
/* ------------------------------------------------------------------ */

static int pluto_render(const char *svg, int w, int h, Image *out) {
    memset(out, 0, sizeof(*out));
    plutosvg_document_t *doc = plutosvg_document_load_from_data(
        svg, (int)strlen(svg), (float)w, (float)h, NULL, NULL);
    if (!doc) return 0;

    plutovg_surface_t *surf = plutosvg_document_render_to_surface(
        doc, NULL, w, h, NULL, NULL, NULL);
    plutosvg_document_destroy(doc);
    if (!surf) return 0;

    int sw = plutovg_surface_get_width(surf);
    int sh = plutovg_surface_get_height(surf);
    int ss = plutovg_surface_get_stride(surf);
    unsigned char *data = plutovg_surface_get_data(surf);

    out->w = sw; out->h = sh;
    out->pixels = (unsigned char *)malloc((size_t)sw * sh * 4);
    plutovg_convert_argb_to_rgba(out->pixels, data, sw, sh, ss);
    plutovg_surface_destroy(surf);
    return 1;
}

/* ------------------------------------------------------------------ */
/*  SigilVG GPU headless renderer                                     */
/* ------------------------------------------------------------------ */

typedef struct {
    WGPUInstance  instance;
    WGPUAdapter   adapter;
    WGPUDevice    device;
    WGPUQueue     queue;
} GPU;

static void on_adapter(WGPURequestAdapterStatus s, WGPUAdapter a,
                        WGPUStringView msg, void *u1, void *u2) {
    (void)s; (void)msg; (void)u2; *(WGPUAdapter *)u1 = a;
}
static void on_device(WGPURequestDeviceStatus s, WGPUDevice d,
                       WGPUStringView msg, void *u1, void *u2) {
    (void)s; (void)msg; (void)u2; *(WGPUDevice *)u1 = d;
}

static GPU g_gpu = {0};
static SigilContext *g_sigil = NULL;

static int gpu_init(void) {
    if (g_gpu.device) return 1; /* already init */

    WGPUInstanceFeatureName feats[] = {
        WGPUInstanceFeatureName_TimedWaitAny,
        WGPUInstanceFeatureName_ShaderSourceSPIRV,
    };
    g_gpu.instance = wgpuCreateInstance(&(WGPUInstanceDescriptor){
        .requiredFeatures = feats, .requiredFeatureCount = 2,
    });
    if (!g_gpu.instance) return 0;

    WGPURequestAdapterCallbackInfo acb = {
        .callback = on_adapter, .mode = WGPUCallbackMode_WaitAnyOnly,
        .userdata1 = &g_gpu.adapter,
    };
    WGPUFuture af = wgpuInstanceRequestAdapter(g_gpu.instance,
        &(WGPURequestAdapterOptions){
            .featureLevel = WGPUFeatureLevel_Compatibility,
            .backendType = WGPUBackendType_WebGPU,
        }, acb);
    wgpuInstanceWaitAny(g_gpu.instance, 1, &(WGPUFutureWaitInfo){.future=af}, 2000000000);
    if (!g_gpu.adapter) return 0;

    WGPURequestDeviceCallbackInfo dcb = {
        .callback = on_device, .mode = WGPUCallbackMode_WaitAnyOnly,
        .userdata1 = &g_gpu.device,
    };
    WGPUFuture df = wgpuAdapterRequestDevice(g_gpu.adapter,
        &(WGPUDeviceDescriptor){0}, dcb);
    wgpuInstanceWaitAny(g_gpu.instance, 1, &(WGPUFutureWaitInfo){.future=df}, 2000000000);
    if (!g_gpu.device) return 0;

    g_gpu.queue = wgpuDeviceGetQueue(g_gpu.device);

    g_sigil = sigil_create(g_gpu.device, WGPUTextureFormat_RGBA8Unorm, 0);
    return g_sigil != NULL;
}

static volatile int g_mapped = 0;
static void map_cb(WGPUMapAsyncStatus status, WGPUStringView msg, void *u1, void *u2) {
    (void)msg; (void)u1; (void)u2;
    g_mapped = (status == WGPUMapAsyncStatus_Success) ? 1 : -1;
}

static int sigil_render(const char *svg, int w, int h, Image *out) {
    memset(out, 0, sizeof(*out));
    if (!gpu_init()) return 0;

    SigilScene *scene = sigil_parse_svg(svg, strlen(svg));
    if (!scene || scene->element_count == 0) { sigil_free_scene(scene); return 0; }

    SigilGPUScene *gpuScene = sigil_upload(g_sigil, scene);
    if (!gpuScene) { sigil_free_scene(scene); return 0; }

    WGPUCommandEncoder prepEnc = wgpuDeviceCreateCommandEncoder(g_gpu.device, NULL);
    SigilDrawData *dd = sigil_prepare_gpu(g_sigil, gpuScene, prepEnc, (float)w, (float)h);
    WGPUCommandBuffer prepCb = wgpuCommandEncoderFinish(prepEnc, NULL);
    wgpuQueueSubmit(g_gpu.queue, 1, &prepCb);
    wgpuCommandBufferRelease(prepCb);
    wgpuCommandEncoderRelease(prepEnc);

    if (!dd) { sigil_free_gpu_scene(gpuScene); sigil_free_scene(scene); return 0; }

    /* Render target */
    WGPUTexture rt = wgpuDeviceCreateTexture(g_gpu.device,
        &(WGPUTextureDescriptor){
            .usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc,
            .dimension = WGPUTextureDimension_2D,
            .size = {(uint32_t)w, (uint32_t)h, 1},
            .format = WGPUTextureFormat_RGBA8Unorm,
            .mipLevelCount = 1, .sampleCount = 1,
        });
    WGPUTextureView rtv = wgpuTextureCreateView(rt,
        &(WGPUTextureViewDescriptor){
            .format = WGPUTextureFormat_RGBA8Unorm,
            .dimension = WGPUTextureViewDimension_2D,
            .mipLevelCount = 1, .arrayLayerCount = 1,
            .aspect = WGPUTextureAspect_All,
            .usage = WGPUTextureUsage_RenderAttachment,
        });

    uint32_t alignedRow = ((uint32_t)w * 4 + 255) & ~255u;
    WGPUBuffer rbuf = wgpuDeviceCreateBuffer(g_gpu.device,
        &(WGPUBufferDescriptor){
            .size = (uint64_t)alignedRow * h,
            .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead,
        });

    /* Pre-fill readback buffer (WGVK doesn't zero-init MapRead buffers) */
    {
        size_t rbSize = (size_t)alignedRow * (size_t)h;
        void *z = calloc(1, rbSize);
        wgpuQueueWriteBuffer(g_gpu.queue, rbuf, 0, z, rbSize);
        free(z);
    }

    /* Encode + submit */
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g_gpu.device, NULL);
    static const float bg[] = {1.0f, 1.0f, 1.0f, 1.0f};
    sigil_encode(g_sigil, dd, enc, rtv, NULL, bg);
    wgpuCommandEncoderCopyTextureToBuffer(enc,
        &(WGPUTexelCopyTextureInfo){.texture = rt, .aspect = WGPUTextureAspect_All},
        &(WGPUTexelCopyBufferInfo){.buffer = rbuf,
            .layout = {.bytesPerRow = alignedRow, .rowsPerImage = (uint32_t)h}},
        &(WGPUExtent3D){(uint32_t)w, (uint32_t)h, 1});
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
    wgpuQueueSubmit(g_gpu.queue, 1, &cb);

    /* Readback */
    g_mapped = 0;
    WGPUFuture mf = wgpuBufferMapAsync(rbuf, WGPUMapMode_Read, 0,
        (size_t)alignedRow * h,
        (WGPUBufferMapCallbackInfo){.callback = map_cb, .mode = WGPUCallbackMode_WaitAnyOnly});
    wgpuInstanceWaitAny(g_gpu.instance, 1, &(WGPUFutureWaitInfo){.future=mf}, UINT64_MAX);

    int ok = 0;
    if (g_mapped == 1) {
        const uint8_t *mapped = (const uint8_t *)wgpuBufferGetMappedRange(
            rbuf, 0, (size_t)alignedRow * h);
        out->w = w; out->h = h;
        out->pixels = (unsigned char *)malloc((size_t)w * h * 4);
        for (int y = 0; y < h; y++)
            memcpy(out->pixels + y * w * 4, mapped + y * alignedRow, (size_t)w * 4);
        ok = 1;
    }

    wgpuBufferUnmap(rbuf);
    wgpuBufferDestroy(rbuf);
    wgpuBufferRelease(rbuf);
    wgpuTextureViewRelease(rtv);
    wgpuTextureDestroy(rt);
    wgpuTextureRelease(rt);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc);
    sigil_free_draw_data(dd);
    sigil_free_gpu_scene(gpuScene);
    sigil_free_scene(scene);
    return ok;
}

/* ------------------------------------------------------------------ */
/*  Comparison helpers                                                */
/* ------------------------------------------------------------------ */

static double image_psnr(const Image *a, const Image *b) {
    if (a->w != b->w || a->h != b->h) return 0.0;
    long long sse = 0;
    int total = a->w * a->h * 4;
    for (int i = 0; i < total; i++) {
        int d = (int)a->pixels[i] - (int)b->pixels[i];
        sse += (long long)(d * d);
    }
    if (sse == 0) return 999.0; /* identical */
    double mse = (double)sse / (double)total;
    return 10.0 * log10(255.0 * 255.0 / mse);
}

static int count_diff_pixels(const Image *a, const Image *b, int threshold) {
    if (a->w != b->w || a->h != b->h) return a->w * a->h;
    int count = 0;
    for (int i = 0; i < a->w * a->h; i++) {
        int off = i * 4;
        int maxd = 0;
        for (int c = 0; c < 4; c++) {
            int d = abs((int)a->pixels[off+c] - (int)b->pixels[off+c]);
            if (d > maxd) maxd = d;
        }
        if (maxd > threshold) count++;
    }
    return count;
}

/* Write a diff visualization: red = differ, green = match */
static void write_diff_image(const Image *a, const Image *b, const char *path, int threshold) {
    if (a->w != b->w || a->h != b->h) return;
    int n = a->w * a->h;
    unsigned char *diff = (unsigned char *)calloc(n * 4, 1);
    for (int i = 0; i < n; i++) {
        int off = i * 4;
        int maxd = 0;
        for (int c = 0; c < 4; c++) {
            int d = abs((int)a->pixels[off+c] - (int)b->pixels[off+c]);
            if (d > maxd) maxd = d;
        }
        if (maxd > threshold) {
            diff[i*4+0] = 255; /* red */
            diff[i*4+3] = 255;
        } else {
            diff[i*4+1] = (unsigned char)(128 + a->pixels[off+1] / 2); /* greenish */
            diff[i*4+3] = 255;
        }
    }
    stbi_write_png(path, a->w, a->h, 4, diff, a->w * 4);
    free(diff);
}

/* ------------------------------------------------------------------ */
/*  The comparison macro: render both, compare, write outputs         */
/* ------------------------------------------------------------------ */

/* PSNR threshold: these are fundamentally different renderers
   (CPU scanline vs GPU analytic curves), so we allow some difference.
   >25 dB is "similar", >30 dB is "good", >35 dB is "great". */
#define MIN_PSNR 20.0

/* Max fraction of pixels that can differ by more than threshold */
#define MAX_DIFF_FRAC 0.15
#define DIFF_THRESHOLD 30

/* Composite RGBA image onto SigilVG's clear color (0.05, 0.05, 0.1, 1.0)
   so PlutoSVG's transparent background matches SigilVG's. */
static void composite_on_clear_color(Image *img) {
    /* SigilVG clear: (0.05, 0.05, 0.1, 1.0) → (13, 13, 26, 255) */
    const unsigned char bgr = 13, bgg = 13, bgb = 26;
    for (int i = 0; i < img->w * img->h; i++) {
        unsigned char *p = img->pixels + i * 4;
        int a = p[3];
        if (a == 255) continue; /* fully opaque — no compositing needed */
        /* premultiplied-alpha over: result = src + (1-src_a) * bg */
        p[0] = (unsigned char)((p[0] * a + bgr * (255 - a)) / 255);
        p[1] = (unsigned char)((p[1] * a + bgg * (255 - a)) / 255);
        p[2] = (unsigned char)((p[2] * a + bgb * (255 - a)) / 255);
        p[3] = 255;
    }
}

static int compare_svg(const char *name, const char *svg, int w, int h) {
    Image ref, gpu;

    if (!pluto_render(svg, w, h, &ref)) {
        printf("[pluto_render FAILED] ");
        return 0;
    }
    /* Composite PlutoSVG output onto same background as SigilVG's clear color */
    composite_on_clear_color(&ref);

    if (!sigil_render(svg, w, h, &gpu)) {
        printf("[sigil_render FAILED] ");
        image_free(&ref);
        return 0;
    }

    double psnr = image_psnr(&ref, &gpu);
    int ndiff = count_diff_pixels(&ref, &gpu, DIFF_THRESHOLD);
    int total = w * h;
    double frac = (double)ndiff / total;

    printf("[PSNR=%.1f dB, diff=%.1f%%] ", psnr, frac * 100.0);

    /* Write outputs for inspection */
    char path[256];
    snprintf(path, sizeof path, "cmp_%s_ref.png", name);
    stbi_write_png(path, ref.w, ref.h, 4, ref.pixels, ref.w * 4);
    snprintf(path, sizeof path, "cmp_%s_gpu.png", name);
    stbi_write_png(path, gpu.w, gpu.h, 4, gpu.pixels, gpu.w * 4);
    snprintf(path, sizeof path, "cmp_%s_diff.png", name);
    write_diff_image(&ref, &gpu, path, DIFF_THRESHOLD);

    image_free(&ref);
    image_free(&gpu);

    return (psnr >= MIN_PSNR && frac <= MAX_DIFF_FRAC);
}

/* ================================================================== */
/*  TEST CASES                                                        */
/* ================================================================== */

/* --- Basic fills --- */

TEST(test_cmp_solid_rect) {
    const char *svg =
        "<svg width=\"128\" height=\"128\" xmlns=\"http://www.w3.org/2000/svg\">"
        "  <rect x=\"0\" y=\"0\" width=\"128\" height=\"128\" fill=\"#ff0000\"/>"
        "</svg>";
    ASSERT(compare_svg("solid_rect", svg, 128, 128));
}

TEST(test_cmp_blue_circle) {
    const char *svg =
        "<svg width=\"128\" height=\"128\" xmlns=\"http://www.w3.org/2000/svg\">"
        "  <circle cx=\"64\" cy=\"64\" r=\"50\" fill=\"blue\"/>"
        "</svg>";
    ASSERT(compare_svg("blue_circle", svg, 128, 128));
}

TEST(test_cmp_green_ellipse) {
    const char *svg =
        "<svg width=\"200\" height=\"100\" xmlns=\"http://www.w3.org/2000/svg\">"
        "  <ellipse cx=\"100\" cy=\"50\" rx=\"80\" ry=\"40\" fill=\"green\"/>"
        "</svg>";
    ASSERT(compare_svg("green_ellipse", svg, 200, 100));
}

TEST(test_cmp_triangle_path) {
    const char *svg =
        "<svg width=\"128\" height=\"128\" xmlns=\"http://www.w3.org/2000/svg\">"
        "  <path d=\"M 64 10 L 120 110 L 8 110 Z\" fill=\"#ff8800\"/>"
        "</svg>";
    ASSERT(compare_svg("triangle", svg, 128, 128));
}

TEST(test_cmp_polygon_star) {
    const char *svg =
        "<svg width=\"200\" height=\"200\" xmlns=\"http://www.w3.org/2000/svg\">"
        "  <polygon points=\"100,10 40,198 190,78 10,78 160,198\" fill=\"gold\"/>"
        "</svg>";
    ASSERT(compare_svg("star", svg, 200, 200));
}

/* --- Path commands --- */

TEST(test_cmp_cubic_bezier) {
    const char *svg =
        "<svg width=\"200\" height=\"200\" xmlns=\"http://www.w3.org/2000/svg\">"
        "  <path d=\"M 10 180 C 30 10, 170 10, 190 180\" "
        "        fill=\"none\" stroke=\"#ff00ff\" stroke-width=\"5\"/>"
        "</svg>";
    ASSERT(compare_svg("cubic", svg, 200, 200));
}

TEST(test_cmp_quadratic_bezier) {
    const char *svg =
        "<svg width=\"200\" height=\"200\" xmlns=\"http://www.w3.org/2000/svg\">"
        "  <path d=\"M 10 180 Q 100 10 190 180\" "
        "        fill=\"none\" stroke=\"cyan\" stroke-width=\"4\"/>"
        "</svg>";
    ASSERT(compare_svg("quadratic", svg, 200, 200));
}

TEST(test_cmp_smooth_cubic) {
    const char *svg =
        "<svg width=\"300\" height=\"150\" xmlns=\"http://www.w3.org/2000/svg\">"
        "  <path d=\"M 10 75 C 30 10 70 10 100 75 S 170 140 200 75 S 270 10 290 75\" "
        "        fill=\"none\" stroke=\"#ff4444\" stroke-width=\"3\"/>"
        "</svg>";
    ASSERT(compare_svg("smooth_cubic", svg, 300, 150));
}

TEST(test_cmp_arc) {
    const char *svg =
        "<svg width=\"200\" height=\"200\" xmlns=\"http://www.w3.org/2000/svg\">"
        "  <path d=\"M 30 100 A 60 60 0 0 1 170 100\" "
        "        fill=\"none\" stroke=\"lime\" stroke-width=\"4\"/>"
        "</svg>";
    ASSERT(compare_svg("arc", svg, 200, 200));
}

TEST(test_cmp_arc_large) {
    const char *svg =
        "<svg width=\"200\" height=\"200\" xmlns=\"http://www.w3.org/2000/svg\">"
        "  <path d=\"M 30 100 A 60 60 0 1 1 170 100\" "
        "        fill=\"#4488cc\" stroke=\"none\"/>"
        "</svg>";
    ASSERT(compare_svg("arc_large", svg, 200, 200));
}

TEST(test_cmp_hv_lines) {
    const char *svg =
        "<svg width=\"100\" height=\"100\" xmlns=\"http://www.w3.org/2000/svg\">"
        "  <path d=\"M 10 10 H 90 V 90 H 10 Z\" fill=\"#884422\"/>"
        "</svg>";
    ASSERT(compare_svg("hv_lines", svg, 100, 100));
}

/* --- Fill rules --- */

TEST(test_cmp_evenodd_star) {
    const char *svg =
        "<svg width=\"200\" height=\"200\" xmlns=\"http://www.w3.org/2000/svg\">"
        "  <polygon points=\"100,10 40,198 190,78 10,78 160,198\" "
        "           fill=\"red\" fill-rule=\"evenodd\"/>"
        "</svg>";
    ASSERT(compare_svg("evenodd_star", svg, 200, 200));
}

TEST(test_cmp_nonzero_star) {
    const char *svg =
        "<svg width=\"200\" height=\"200\" xmlns=\"http://www.w3.org/2000/svg\">"
        "  <polygon points=\"100,10 40,198 190,78 10,78 160,198\" "
        "           fill=\"blue\" fill-rule=\"nonzero\"/>"
        "</svg>";
    ASSERT(compare_svg("nonzero_star", svg, 200, 200));
}

/* --- Strokes --- */

TEST(test_cmp_stroke_line) {
    const char *svg =
        "<svg width=\"200\" height=\"100\" xmlns=\"http://www.w3.org/2000/svg\">"
        "  <line x1=\"10\" y1=\"50\" x2=\"190\" y2=\"50\" stroke=\"red\" stroke-width=\"6\"/>"
        "</svg>";
    ASSERT(compare_svg("stroke_line", svg, 200, 100));
}

TEST(test_cmp_stroke_rect) {
    const char *svg =
        "<svg width=\"150\" height=\"150\" xmlns=\"http://www.w3.org/2000/svg\">"
        "  <rect x=\"20\" y=\"20\" width=\"110\" height=\"110\" "
        "        fill=\"none\" stroke=\"#00ccff\" stroke-width=\"4\"/>"
        "</svg>";
    ASSERT(compare_svg("stroke_rect", svg, 150, 150));
}

TEST(test_cmp_fill_and_stroke) {
    const char *svg =
        "<svg width=\"150\" height=\"150\" xmlns=\"http://www.w3.org/2000/svg\">"
        "  <circle cx=\"75\" cy=\"75\" r=\"50\" fill=\"#336699\" stroke=\"white\" stroke-width=\"3\"/>"
        "</svg>";
    ASSERT(compare_svg("fill_stroke", svg, 150, 150));
}

TEST(test_cmp_polyline_stroke) {
    const char *svg =
        "<svg width=\"200\" height=\"100\" xmlns=\"http://www.w3.org/2000/svg\">"
        "  <polyline points=\"10,80 50,20 90,80 130,20 170,80\" "
        "            fill=\"none\" stroke=\"orange\" stroke-width=\"3\"/>"
        "</svg>";
    ASSERT(compare_svg("polyline_stroke", svg, 200, 100));
}

/* --- Transforms --- */

TEST(test_cmp_translate) {
    const char *svg =
        "<svg width=\"200\" height=\"200\" xmlns=\"http://www.w3.org/2000/svg\">"
        "  <g transform=\"translate(50,50)\">"
        "    <rect x=\"0\" y=\"0\" width=\"100\" height=\"100\" fill=\"yellow\"/>"
        "  </g>"
        "</svg>";
    ASSERT(compare_svg("translate", svg, 200, 200));
}

TEST(test_cmp_scale) {
    const char *svg =
        "<svg width=\"200\" height=\"200\" xmlns=\"http://www.w3.org/2000/svg\">"
        "  <g transform=\"scale(2)\">"
        "    <rect x=\"10\" y=\"10\" width=\"50\" height=\"50\" fill=\"#cc44cc\"/>"
        "  </g>"
        "</svg>";
    ASSERT(compare_svg("scale", svg, 200, 200));
}

TEST(test_cmp_rotate) {
    const char *svg =
        "<svg width=\"200\" height=\"200\" xmlns=\"http://www.w3.org/2000/svg\">"
        "  <g transform=\"rotate(45,100,100)\">"
        "    <rect x=\"60\" y=\"60\" width=\"80\" height=\"80\" fill=\"#44cc88\"/>"
        "  </g>"
        "</svg>";
    ASSERT(compare_svg("rotate", svg, 200, 200));
}

TEST(test_cmp_nested_transforms) {
    const char *svg =
        "<svg width=\"200\" height=\"200\" xmlns=\"http://www.w3.org/2000/svg\">"
        "  <g transform=\"translate(100,100)\">"
        "    <g transform=\"rotate(30)\">"
        "      <rect x=\"-40\" y=\"-40\" width=\"80\" height=\"80\" fill=\"coral\"/>"
        "    </g>"
        "  </g>"
        "</svg>";
    ASSERT(compare_svg("nested_xform", svg, 200, 200));
}

TEST(test_cmp_matrix_transform) {
    const char *svg =
        "<svg width=\"200\" height=\"200\" xmlns=\"http://www.w3.org/2000/svg\">"
        "  <g transform=\"matrix(0.866,0.5,-0.5,0.866,100,100)\">"
        "    <rect x=\"-30\" y=\"-30\" width=\"60\" height=\"60\" fill=\"#8844cc\"/>"
        "  </g>"
        "</svg>";
    ASSERT(compare_svg("matrix_xform", svg, 200, 200));
}

/* --- Opacity --- */

TEST(test_cmp_opacity) {
    const char *svg =
        "<svg width=\"200\" height=\"200\" xmlns=\"http://www.w3.org/2000/svg\">"
        "  <rect x=\"0\" y=\"0\" width=\"200\" height=\"200\" fill=\"#222222\"/>"
        "  <circle cx=\"100\" cy=\"100\" r=\"80\" fill=\"red\" opacity=\"0.5\"/>"
        "</svg>";
    ASSERT(compare_svg("opacity", svg, 200, 200));
}

TEST(test_cmp_overlapping_opacity) {
    const char *svg =
        "<svg width=\"200\" height=\"200\" xmlns=\"http://www.w3.org/2000/svg\">"
        "  <rect x=\"0\" y=\"0\" width=\"200\" height=\"200\" fill=\"white\"/>"
        "  <rect x=\"20\" y=\"20\" width=\"120\" height=\"120\" fill=\"red\" opacity=\"0.5\"/>"
        "  <rect x=\"60\" y=\"60\" width=\"120\" height=\"120\" fill=\"blue\" opacity=\"0.5\"/>"
        "</svg>";
    ASSERT(compare_svg("overlap_opacity", svg, 200, 200));
}

/* --- viewBox --- */

TEST(test_cmp_viewbox_scale_up) {
    const char *svg =
        "<svg viewBox=\"0 0 50 50\" width=\"200\" height=\"200\" xmlns=\"http://www.w3.org/2000/svg\">"
        "  <circle cx=\"25\" cy=\"25\" r=\"20\" fill=\"#ff6600\"/>"
        "</svg>";
    ASSERT(compare_svg("viewbox_up", svg, 200, 200));
}

TEST(test_cmp_viewbox_scale_down) {
    const char *svg =
        "<svg viewBox=\"0 0 400 400\" width=\"100\" height=\"100\" xmlns=\"http://www.w3.org/2000/svg\">"
        "  <rect x=\"50\" y=\"50\" width=\"300\" height=\"300\" fill=\"teal\"/>"
        "</svg>";
    ASSERT(compare_svg("viewbox_down", svg, 100, 100));
}

/* --- Colors --- */

TEST(test_cmp_named_colors) {
    const char *svg =
        "<svg width=\"300\" height=\"50\" xmlns=\"http://www.w3.org/2000/svg\">"
        "  <rect x=\"0\"   y=\"0\" width=\"50\" height=\"50\" fill=\"red\"/>"
        "  <rect x=\"50\"  y=\"0\" width=\"50\" height=\"50\" fill=\"green\"/>"
        "  <rect x=\"100\" y=\"0\" width=\"50\" height=\"50\" fill=\"blue\"/>"
        "  <rect x=\"150\" y=\"0\" width=\"50\" height=\"50\" fill=\"yellow\"/>"
        "  <rect x=\"200\" y=\"0\" width=\"50\" height=\"50\" fill=\"purple\"/>"
        "  <rect x=\"250\" y=\"0\" width=\"50\" height=\"50\" fill=\"orange\"/>"
        "</svg>";
    ASSERT(compare_svg("named_colors", svg, 300, 50));
}

TEST(test_cmp_hex_colors) {
    const char *svg =
        "<svg width=\"200\" height=\"50\" xmlns=\"http://www.w3.org/2000/svg\">"
        "  <rect x=\"0\"   y=\"0\" width=\"50\" height=\"50\" fill=\"#f00\"/>"
        "  <rect x=\"50\"  y=\"0\" width=\"50\" height=\"50\" fill=\"#00ff00\"/>"
        "  <rect x=\"100\" y=\"0\" width=\"50\" height=\"50\" fill=\"#0000ff\"/>"
        "  <rect x=\"150\" y=\"0\" width=\"50\" height=\"50\" fill=\"#abcdef\"/>"
        "</svg>";
    ASSERT(compare_svg("hex_colors", svg, 200, 50));
}

/* --- Composition --- */

TEST(test_cmp_multi_shapes) {
    const char *svg =
        "<svg width=\"200\" height=\"200\" xmlns=\"http://www.w3.org/2000/svg\">"
        "  <rect x=\"0\" y=\"0\" width=\"200\" height=\"200\" fill=\"#1a1a2e\"/>"
        "  <circle cx=\"100\" cy=\"100\" r=\"80\" fill=\"#e94560\"/>"
        "  <rect x=\"60\" y=\"60\" width=\"80\" height=\"80\" fill=\"#0f3460\"/>"
        "</svg>";
    ASSERT(compare_svg("multi_shapes", svg, 200, 200));
}

TEST(test_cmp_complex_scene) {
    const char *svg =
        "<svg viewBox=\"0 0 400 400\" xmlns=\"http://www.w3.org/2000/svg\">"
        "  <rect x=\"0\" y=\"0\" width=\"400\" height=\"400\" fill=\"#1a1a2e\"/>"
        "  <g transform=\"translate(200,200)\">"
        "    <circle cx=\"0\" cy=\"0\" r=\"150\" fill=\"#16213e\"/>"
        "    <ellipse cx=\"0\" cy=\"20\" rx=\"100\" ry=\"80\" fill=\"#0f3460\"/>"
        "    <polygon points=\"-30,-60 0,-90 30,-60\" fill=\"#e94560\"/>"
        "    <line x1=\"-60\" y1=\"40\" x2=\"60\" y2=\"40\" stroke=\"#ffffff\" stroke-width=\"2\"/>"
        "  </g>"
        "  <rect x=\"10\" y=\"10\" width=\"80\" height=\"30\" fill=\"#e94560\" opacity=\"0.7\"/>"
        "</svg>";
    ASSERT(compare_svg("complex_scene", svg, 400, 400));
}

TEST(test_cmp_circles_grid) {
    const char *svg =
        "<svg width=\"300\" height=\"300\" xmlns=\"http://www.w3.org/2000/svg\">"
        "  <rect x=\"0\" y=\"0\" width=\"300\" height=\"300\" fill=\"#111\"/>"
        "  <circle cx=\"75\"  cy=\"75\"  r=\"50\" fill=\"red\"/>"
        "  <circle cx=\"225\" cy=\"75\"  r=\"50\" fill=\"green\"/>"
        "  <circle cx=\"75\"  cy=\"225\" r=\"50\" fill=\"blue\"/>"
        "  <circle cx=\"225\" cy=\"225\" r=\"50\" fill=\"yellow\"/>"
        "  <circle cx=\"150\" cy=\"150\" r=\"60\" fill=\"white\" opacity=\"0.3\"/>"
        "</svg>";
    ASSERT(compare_svg("circles_grid", svg, 300, 300));
}

/* --- Edge cases --- */

TEST(test_cmp_tiny_shape) {
    const char *svg =
        "<svg width=\"100\" height=\"100\" xmlns=\"http://www.w3.org/2000/svg\">"
        "  <circle cx=\"50\" cy=\"50\" r=\"2\" fill=\"white\"/>"
        "</svg>";
    ASSERT(compare_svg("tiny_shape", svg, 100, 100));
}

TEST(test_cmp_large_circle) {
    const char *svg =
        "<svg width=\"200\" height=\"200\" xmlns=\"http://www.w3.org/2000/svg\">"
        "  <circle cx=\"100\" cy=\"100\" r=\"99\" fill=\"#336699\"/>"
        "</svg>";
    ASSERT(compare_svg("large_circle", svg, 200, 200));
}

TEST(test_cmp_concentric_circles) {
    const char *svg =
        "<svg width=\"200\" height=\"200\" xmlns=\"http://www.w3.org/2000/svg\">"
        "  <circle cx=\"100\" cy=\"100\" r=\"90\" fill=\"#001133\"/>"
        "  <circle cx=\"100\" cy=\"100\" r=\"70\" fill=\"#002266\"/>"
        "  <circle cx=\"100\" cy=\"100\" r=\"50\" fill=\"#003399\"/>"
        "  <circle cx=\"100\" cy=\"100\" r=\"30\" fill=\"#0044cc\"/>"
        "  <circle cx=\"100\" cy=\"100\" r=\"10\" fill=\"#0055ff\"/>"
        "</svg>";
    ASSERT(compare_svg("concentric", svg, 200, 200));
}

TEST(test_cmp_adjacent_rects) {
    const char *svg =
        "<svg width=\"200\" height=\"100\" xmlns=\"http://www.w3.org/2000/svg\">"
        "  <rect x=\"0\"   y=\"0\" width=\"50\"  height=\"100\" fill=\"#ff0000\"/>"
        "  <rect x=\"50\"  y=\"0\" width=\"50\"  height=\"100\" fill=\"#00ff00\"/>"
        "  <rect x=\"100\" y=\"0\" width=\"50\"  height=\"100\" fill=\"#0000ff\"/>"
        "  <rect x=\"150\" y=\"0\" width=\"50\"  height=\"100\" fill=\"#ffff00\"/>"
        "</svg>";
    ASSERT(compare_svg("adjacent_rects", svg, 200, 100));
}

/* --- Closed path shapes --- */

TEST(test_cmp_diamond) {
    const char *svg =
        "<svg width=\"200\" height=\"200\" xmlns=\"http://www.w3.org/2000/svg\">"
        "  <path d=\"M 100 10 L 190 100 L 100 190 L 10 100 Z\" fill=\"#cc00cc\"/>"
        "</svg>";
    ASSERT(compare_svg("diamond", svg, 200, 200));
}

TEST(test_cmp_heart) {
    const char *svg =
        "<svg width=\"200\" height=\"200\" xmlns=\"http://www.w3.org/2000/svg\">"
        "  <path d=\"M 100 180 "
        "           C 40 120 0 80 0 50 "
        "           C 0 20 30 0 55 0 "
        "           C 80 0 100 20 100 40 "
        "           C 100 20 120 0 145 0 "
        "           C 170 0 200 20 200 50 "
        "           C 200 80 160 120 100 180 Z\" fill=\"#ff2255\"/>"
        "</svg>";
    ASSERT(compare_svg("heart", svg, 200, 200));
}

/* ------------------------------------------------------------------ */
/*  Main                                                              */
/* ------------------------------------------------------------------ */

int main(void) {
    printf("SigilVG vs PlutoSVG comparison tests\n");
    printf("  PlutoSVG %s / PlutoVG %s\n\n", plutosvg_version_string(), plutovg_version_string());

    /* Init GPU once */
    if (!gpu_init()) {
        fprintf(stderr, "ERROR: GPU init failed, cannot run comparison tests\n");
        return 1;
    }
    printf("  GPU initialized\n\n");

    printf("-- Basic fills --\n");
    RUN(test_cmp_solid_rect);
    RUN(test_cmp_blue_circle);
    RUN(test_cmp_green_ellipse);
    RUN(test_cmp_triangle_path);
    RUN(test_cmp_polygon_star);

    printf("\n-- Path commands --\n");
    RUN(test_cmp_cubic_bezier);
    RUN(test_cmp_quadratic_bezier);
    RUN(test_cmp_smooth_cubic);
    RUN(test_cmp_arc);
    RUN(test_cmp_arc_large);
    RUN(test_cmp_hv_lines);

    printf("\n-- Fill rules --\n");
    RUN(test_cmp_evenodd_star);
    RUN(test_cmp_nonzero_star);

    printf("\n-- Strokes --\n");
    RUN(test_cmp_stroke_line);
    RUN(test_cmp_stroke_rect);
    RUN(test_cmp_fill_and_stroke);
    RUN(test_cmp_polyline_stroke);

    printf("\n-- Transforms --\n");
    RUN(test_cmp_translate);
    RUN(test_cmp_scale);
    RUN(test_cmp_rotate);
    RUN(test_cmp_nested_transforms);
    RUN(test_cmp_matrix_transform);

    printf("\n-- Opacity --\n");
    RUN(test_cmp_opacity);
    RUN(test_cmp_overlapping_opacity);

    printf("\n-- viewBox --\n");
    RUN(test_cmp_viewbox_scale_up);
    RUN(test_cmp_viewbox_scale_down);

    printf("\n-- Colors --\n");
    RUN(test_cmp_named_colors);
    RUN(test_cmp_hex_colors);

    printf("\n-- Composition --\n");
    RUN(test_cmp_multi_shapes);
    RUN(test_cmp_complex_scene);
    RUN(test_cmp_circles_grid);

    printf("\n-- Edge cases --\n");
    RUN(test_cmp_tiny_shape);
    RUN(test_cmp_large_circle);
    RUN(test_cmp_concentric_circles);
    RUN(test_cmp_adjacent_rects);

    printf("\n-- Complex shapes --\n");
    RUN(test_cmp_diamond);
    RUN(test_cmp_heart);

    printf("\n%d passed, %d failed\n", g_pass, g_fail);

    if (g_sigil) sigil_destroy(g_sigil);
    return g_fail > 0 ? 1 : 0;
}
