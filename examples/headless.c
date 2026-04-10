/*
 * SigilVG headless renderer — renders SVG to an offscreen texture and saves PNG.
 * No window, no surface, no swapchain.  Uses only webgpu.h — no GLFW.
 *
 * Usage: sigilvg_headless [input.svg] [output.png]
 *   If no input SVG given, renders a built-in test scene.
 *   If no output path given, writes "output.png".
 */

#define SIGIL_IMPLEMENTATION
#include "sigilvg.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Default SVG for testing                                           */
/* ------------------------------------------------------------------ */

static const char *DEFAULT_SVG =
    "<svg viewBox=\"0 0 200 200\">"
    "  <rect x=\"10\" y=\"10\" width=\"180\" height=\"180\" fill=\"#2244aa\"/>"
    "  <circle cx=\"100\" cy=\"100\" r=\"60\" fill=\"#ffcc00\"/>"
    "  <path d=\"M 50 150 L 100 50 L 150 150 Z\" fill=\"#ff4444\" fill-rule=\"evenodd\"/>"
    "</svg>";

/* ------------------------------------------------------------------ */
/*  Headless WebGPU init                                              */
/* ------------------------------------------------------------------ */

typedef struct {
    WGPUInstance  instance;
    WGPUAdapter   adapter;
    WGPUDevice    device;
    WGPUQueue     queue;
} HeadlessGPU;

static void on_adapter(WGPURequestAdapterStatus s, WGPUAdapter a,
                        WGPUStringView msg, void *u1, void *u2) {
    (void)s; (void)msg; (void)u2;
    *(WGPUAdapter *)u1 = a;
}
static void on_device(WGPURequestDeviceStatus s, WGPUDevice d,
                       WGPUStringView msg, void *u1, void *u2) {
    (void)s; (void)msg; (void)u2;
    *(WGPUDevice *)u1 = d;
}

static HeadlessGPU headless_gpu_init(void)
{
    HeadlessGPU g = {0};

    WGPUInstanceFeatureName features[] = {
        WGPUInstanceFeatureName_TimedWaitAny,
        WGPUInstanceFeatureName_ShaderSourceSPIRV,
    };
    WGPUInstanceDescriptor idesc = {
        .requiredFeatures     = features,
        .requiredFeatureCount = 2,
    };
    g.instance = wgpuCreateInstance(&idesc);
    if (!g.instance) {
        fprintf(stderr, "Failed to create WebGPU instance\n");
        return g;
    }

    /* Adapter */
    WGPURequestAdapterOptions aopts = {
        .featureLevel = WGPUFeatureLevel_Compatibility,
        .backendType  = WGPUBackendType_WebGPU,
    };
    WGPURequestAdapterCallbackInfo acb = {
        .callback  = on_adapter,
        .mode      = WGPUCallbackMode_WaitAnyOnly,
        .userdata1 = &g.adapter,
    };
    WGPUFuture af = wgpuInstanceRequestAdapter(g.instance, &aopts, acb);
    WGPUFutureWaitInfo aw = { .future = af };
    wgpuInstanceWaitAny(g.instance, 1, &aw, 1000000000);
    if (!g.adapter) {
        fprintf(stderr, "Failed to get WebGPU adapter\n");
        return g;
    }

    /* Device */
    WGPUDeviceDescriptor ddesc = {0};
    WGPURequestDeviceCallbackInfo dcb = {
        .callback  = on_device,
        .mode      = WGPUCallbackMode_WaitAnyOnly,
        .userdata1 = &g.device,
    };
    WGPUFuture df = wgpuAdapterRequestDevice(g.adapter, &ddesc, dcb);
    WGPUFutureWaitInfo dw = { .future = df };
    wgpuInstanceWaitAny(g.instance, 1, &dw, 1000000000);
    if (!g.device) {
        fprintf(stderr, "Failed to get WebGPU device\n");
        return g;
    }

    g.queue = wgpuDeviceGetQueue(g.device);
    return g;
}

/* ------------------------------------------------------------------ */
/*  File reading helper                                               */
/* ------------------------------------------------------------------ */

static char *read_file(const char *path, size_t *out_size)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Error: cannot open '%s'\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)sz + 1);
    size_t rd = fread(buf, 1, (size_t)sz, f);
    buf[rd] = '\0';
    fclose(f);
    if (out_size) *out_size = rd;
    return buf;
}

/* ------------------------------------------------------------------ */
/*  Buffer map callback                                               */
/* ------------------------------------------------------------------ */

static volatile int g_mapped = 0;
static void map_cb(WGPUMapAsyncStatus status, WGPUStringView msg, void *u1, void *u2) {
    (void)msg; (void)u1; (void)u2;
    g_mapped = (status == WGPUMapAsyncStatus_Success) ? 1 : -1;
}

/* ------------------------------------------------------------------ */
/*  Main                                                              */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    const char *inputPath = NULL;
    const char *outPath   = "output.png";
    int reqW = 0, reqH = 0;

    if (argc >= 2) inputPath = argv[1];
    if (argc >= 3) outPath   = argv[2];
    if (argc >= 5) { reqW = atoi(argv[3]); reqH = atoi(argv[4]); }

    /* Load or use default SVG */
    char *svgBuf = NULL;
    const char *svgData;
    size_t svgLen;

    if (inputPath) {
        svgBuf = read_file(inputPath, &svgLen);
        if (!svgBuf) return 1;
        svgData = svgBuf;
        fprintf(stderr, "Input:  %s (%zu bytes)\n", inputPath, svgLen);
    } else {
        svgData = DEFAULT_SVG;
        svgLen = strlen(DEFAULT_SVG);
        fprintf(stderr, "Input:  <built-in test SVG>\n");
    }
    fprintf(stderr, "Output: %s\n", outPath);

    /* Parse SVG */
    SigilScene *scene = sigil_parse_svg(svgData, svgLen);
    free(svgBuf);

    if (!scene || scene->element_count == 0) {
        fprintf(stderr, "Error: no renderable elements found in SVG\n");
        sigil_free_scene(scene);
        return 1;
    }
    fprintf(stderr, "Parsed: %d element(s)\n", scene->element_count);

    /* Image dimensions */
    const int W = (reqW > 0) ? reqW : 512;
    const int H = (reqH > 0) ? reqH : 512;
    fprintf(stderr, "Image:  %dx%d\n", W, H);

    /* Headless GPU init — no window, no surface */
    HeadlessGPU gpu = headless_gpu_init();
    if (!gpu.device) {
        fprintf(stderr, "Headless GPU init failed\n");
        sigil_free_scene(scene);
        return 1;
    }

    /* Create SigilVG pipeline context */
    SigilContext *ctx = sigil_create(gpu.device, WGPUTextureFormat_RGBA8Unorm, 0);
    if (!ctx) {
        fprintf(stderr, "Error: sigil_create failed (check shader files)\n");
        sigil_free_scene(scene);
        return 1;
    }

    /* Prepare draw data (band building, texture packing, GPU upload) */
    SigilDrawData *dd = sigil_prepare(ctx, scene, (float)W, (float)H, false);
    if (!dd) {
        fprintf(stderr, "Error: sigil_prepare returned NULL (no geometry?)\n");
        sigil_destroy(ctx);
        sigil_free_scene(scene);
        return 1;
    }

    /* Create offscreen render target */
    WGPUTexture rtTex = wgpuDeviceCreateTexture(gpu.device,
        &(WGPUTextureDescriptor){
            .usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc,
            .dimension = WGPUTextureDimension_2D,
            .size = {(uint32_t)W, (uint32_t)H, 1},
            .format = WGPUTextureFormat_RGBA8Unorm,
            .mipLevelCount = 1, .sampleCount = 1,
        });
    WGPUTextureView rtView = wgpuTextureCreateView(rtTex,
        &(WGPUTextureViewDescriptor){
            .format = WGPUTextureFormat_RGBA8Unorm,
            .dimension = WGPUTextureViewDimension_2D,
            .mipLevelCount = 1, .arrayLayerCount = 1,
            .aspect = WGPUTextureAspect_All,
            .usage = WGPUTextureUsage_RenderAttachment,
        });

    /* Readback buffer (RGBA8, 4 bytes per pixel, 256-byte row alignment) */
    uint32_t rowBytes   = (uint32_t)W * 4;
    uint32_t alignedRow = (rowBytes + 255) & ~255u;
    WGPUBuffer readBuf  = wgpuDeviceCreateBuffer(gpu.device,
        &(WGPUBufferDescriptor){
            .size  = (uint64_t)alignedRow * (uint64_t)H,
            .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead,
        });

    static const float bg[] = {1.0f, 1.0f, 1.0f, 1.0f};

    /* Stress test: 200 submits in one frame (well past old 100 semaphore limit) */
    const int STRESS_SUBMITS = 200;
    fprintf(stderr, "Stress test: %d queue submits in one frame...\n", STRESS_SUBMITS);
    for (int i = 0; i < STRESS_SUBMITS; i++) {
        WGPUCommandEncoder e = wgpuDeviceCreateCommandEncoder(gpu.device, NULL);
        sigil_encode(ctx, dd, e, rtView, NULL, bg);
        WGPUCommandBuffer c = wgpuCommandEncoderFinish(e, NULL);
        wgpuQueueSubmit(gpu.queue, 1, &c);
        wgpuCommandBufferRelease(c);
        wgpuCommandEncoderRelease(e);
    }
    fprintf(stderr, "Stress test passed (%d submits).\n", STRESS_SUBMITS);

    /* Final render + copy to readback buffer */
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(gpu.device, NULL);
    sigil_encode(ctx, dd, enc, rtView, NULL, bg);

    wgpuCommandEncoderCopyTextureToBuffer(enc,
        &(WGPUTexelCopyTextureInfo){.texture = rtTex, .aspect = WGPUTextureAspect_All},
        &(WGPUTexelCopyBufferInfo){.buffer = readBuf,
            .layout = {.bytesPerRow = alignedRow, .rowsPerImage = (uint32_t)H}},
        &(WGPUExtent3D){(uint32_t)W, (uint32_t)H, 1});

    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
    wgpuQueueSubmit(gpu.queue, 1, &cb);

    /* Map and read back */
    WGPUBufferMapCallbackInfo mapInfo = {
        .callback = map_cb,
        .mode     = WGPUCallbackMode_WaitAnyOnly,
    };
    WGPUFuture mapFuture = wgpuBufferMapAsync(readBuf, WGPUMapMode_Read,
                                               0, (size_t)alignedRow * H, mapInfo);
    WGPUFutureWaitInfo waitInfo = { .future = mapFuture };
    wgpuInstanceWaitAny(gpu.instance, 1, &waitInfo, UINT64_MAX);

    if (g_mapped != 1) {
        fprintf(stderr, "Error: buffer map failed\n");
        sigil_free_draw_data(dd);
        sigil_free_scene(scene);
        sigil_destroy(ctx);
        return 1;
    }

    const uint8_t *mapped = (const uint8_t *)wgpuBufferGetMappedRange(
        readBuf, 0, (size_t)alignedRow * H);

    /* Build tightly packed RGBA buffer (stb_image_write needs contiguous rows) */
    uint8_t *rgba = (uint8_t *)malloc((size_t)W * (size_t)H * 4);
    for (int y = 0; y < H; y++)
        memcpy(rgba + y * W * 4, mapped + y * alignedRow, (size_t)W * 4);

    stbi_write_png(outPath, W, H, 4, rgba, W * 4);
    free(rgba);
    printf("Wrote %s (%dx%d)\n", outPath, W, H);

    /* Cleanup */
    wgpuBufferUnmap(readBuf);
    wgpuBufferDestroy(readBuf);
    wgpuBufferRelease(readBuf);
    wgpuTextureViewRelease(rtView);
    wgpuTextureDestroy(rtTex);
    wgpuTextureRelease(rtTex);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc);

    sigil_free_draw_data(dd);
    sigil_free_scene(scene);
    sigil_destroy(ctx);

    return 0;
}
