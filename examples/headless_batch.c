/*
 * SigilVG batch headless renderer.
 * Initializes GPU + pipeline ONCE, then processes many SVGs.
 *
 * Usage:
 *   sigilvg_headless_batch <file_list.txt> [output_dir] [width] [height]
 *
 *   file_list.txt: one SVG path per line
 *   output_dir:    where to write PNGs (default: "batch_out")
 *   width/height:  render dimensions (default: 300x300)
 *
 * Writes one PNG per SVG. Prints status per file to stdout.
 */

#define SIGIL_IMPLEMENTATION
#include "sigilvg.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <setjmp.h>
#include <unistd.h>

/* ------------------------------------------------------------------ */
/*  Crash recovery via setjmp/longjmp                                 */
/* ------------------------------------------------------------------ */

static sigjmp_buf g_jmpbuf;
static volatile sig_atomic_t g_in_render = 0;

static void crash_handler(int sig) {
    (void)sig;
    if (g_in_render)
        siglongjmp(g_jmpbuf, 1);
    _exit(128 + sig);
}

/* ------------------------------------------------------------------ */
/*  GPU init (once)                                                   */
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

static GPU gpu_init(void) {
    GPU g = {0};
    WGPUInstanceFeatureName feats[] = {
        WGPUInstanceFeatureName_TimedWaitAny,
        WGPUInstanceFeatureName_ShaderSourceSPIRV,
    };
    g.instance = wgpuCreateInstance(&(WGPUInstanceDescriptor){
        .requiredFeatures = feats, .requiredFeatureCount = 2,
    });
    if (!g.instance) { fprintf(stderr, "wgpuCreateInstance failed\n"); return g; }

    WGPURequestAdapterCallbackInfo acb = {
        .callback = on_adapter, .mode = WGPUCallbackMode_WaitAnyOnly,
        .userdata1 = &g.adapter,
    };
    WGPUFuture af = wgpuInstanceRequestAdapter(g.instance,
        &(WGPURequestAdapterOptions){
            .featureLevel = WGPUFeatureLevel_Compatibility,
            .backendType = WGPUBackendType_WebGPU,
        }, acb);
    wgpuInstanceWaitAny(g.instance, 1, &(WGPUFutureWaitInfo){.future=af}, 2000000000);
    if (!g.adapter) { fprintf(stderr, "Adapter request failed\n"); return g; }

    WGPURequestDeviceCallbackInfo dcb = {
        .callback = on_device, .mode = WGPUCallbackMode_WaitAnyOnly,
        .userdata1 = &g.device,
    };
    WGPUFuture df = wgpuAdapterRequestDevice(g.adapter,
        &(WGPUDeviceDescriptor){0}, dcb);
    wgpuInstanceWaitAny(g.instance, 1, &(WGPUFutureWaitInfo){.future=df}, 2000000000);
    if (!g.device) { fprintf(stderr, "Device request failed\n"); return g; }

    g.queue = wgpuDeviceGetQueue(g.device);
    return g;
}

/* ------------------------------------------------------------------ */
/*  File helpers                                                      */
/* ------------------------------------------------------------------ */

static char *read_file(const char *path, size_t *out_size) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
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

/* Extract basename without extension */
static void get_stem(const char *path, char *out, int maxlen) {
    const char *slash = strrchr(path, '/');
    const char *base = slash ? slash + 1 : path;
    const char *dot = strrchr(base, '.');
    int len = dot ? (int)(dot - base) : (int)strlen(base);
    if (len >= maxlen) len = maxlen - 1;
    memcpy(out, base, len);
    out[len] = '\0';
}

/* ------------------------------------------------------------------ */
/*  Render one SVG to RGBA pixels                                     */
/* ------------------------------------------------------------------ */

static volatile int g_mapped = 0;
static void map_cb(WGPUMapAsyncStatus status, WGPUStringView msg, void *u1, void *u2) {
    (void)msg; (void)u1; (void)u2;
    g_mapped = (status == WGPUMapAsyncStatus_Success) ? 1 : -1;
}

/* Returns malloc'd RGBA pixels, caller frees. NULL on failure. */
static unsigned char *render_one(GPU *gpu, SigilContext *ctx,
                                  const char *svg_data, size_t svg_len,
                                  int W, int H)
{
    SigilScene *scene = sigil_parse_svg(svg_data, svg_len);
    if (!scene || scene->element_count == 0) {
        sigil_free_scene(scene);
        return NULL;
    }

    SigilGPUScene *gpuScene = sigil_upload(ctx, scene);
    if (!gpuScene) { sigil_free_scene(scene); return NULL; }

    WGPUCommandEncoder prepEnc = wgpuDeviceCreateCommandEncoder(gpu->device, NULL);
    SigilDrawData *dd = sigil_prepare_gpu(ctx, gpuScene, prepEnc, (float)W, (float)H);
    WGPUCommandBuffer prepCb = wgpuCommandEncoderFinish(prepEnc, NULL);
    wgpuQueueSubmit(gpu->queue, 1, &prepCb);
    wgpuCommandBufferRelease(prepCb);
    wgpuCommandEncoderRelease(prepEnc);

    if (!dd) { sigil_free_gpu_scene(gpuScene); sigil_free_scene(scene); return NULL; }

    /* Render target */
    WGPUTexture rt = wgpuDeviceCreateTexture(gpu->device,
        &(WGPUTextureDescriptor){
            .usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc,
            .dimension = WGPUTextureDimension_2D,
            .size = {(uint32_t)W, (uint32_t)H, 1},
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

    uint32_t alignedRow = ((uint32_t)W * 4 + 255) & ~255u;
    WGPUBuffer rbuf = wgpuDeviceCreateBuffer(gpu->device,
        &(WGPUBufferDescriptor){
            .size = (uint64_t)alignedRow * H,
            .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead,
        });

    /* Encode + submit */
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(gpu->device, NULL);
    static const float bg[] = {1.0f, 1.0f, 1.0f, 1.0f};
    sigil_encode(ctx, dd, enc, rtv, NULL, bg);

    wgpuCommandEncoderCopyTextureToBuffer(enc,
        &(WGPUTexelCopyTextureInfo){.texture = rt, .aspect = WGPUTextureAspect_All},
        &(WGPUTexelCopyBufferInfo){.buffer = rbuf,
            .layout = {.bytesPerRow = alignedRow, .rowsPerImage = (uint32_t)H}},
        &(WGPUExtent3D){(uint32_t)W, (uint32_t)H, 1});

    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
    wgpuQueueSubmit(gpu->queue, 1, &cb);

    /* Readback */
    g_mapped = 0;
    WGPUFuture mf = wgpuBufferMapAsync(rbuf, WGPUMapMode_Read, 0,
        (size_t)alignedRow * H,
        (WGPUBufferMapCallbackInfo){.callback = map_cb, .mode = WGPUCallbackMode_WaitAnyOnly});
    wgpuInstanceWaitAny(gpu->instance, 1, &(WGPUFutureWaitInfo){.future=mf}, UINT64_MAX);

    unsigned char *pixels = NULL;
    if (g_mapped == 1) {
        const uint8_t *mapped = (const uint8_t *)wgpuBufferGetMappedRange(
            rbuf, 0, (size_t)alignedRow * H);
        pixels = (unsigned char *)malloc((size_t)W * H * 4);
        for (int y = 0; y < H; y++)
            memcpy(pixels + y * W * 4, mapped + y * alignedRow, (size_t)W * 4);
    }

    /* Cleanup GPU objects for this frame */
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

    /* Advance frame — resets WGVK's per-frame submit counter (max 100 submits per frame) */
    wgpuDeviceTick(gpu->device);

    return pixels;
}

/* ------------------------------------------------------------------ */
/*  Main                                                              */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <file_list.txt> [output_dir] [width] [height]\n"
            "  file_list.txt: one SVG path per line\n"
            "  output_dir:    default \"batch_out\"\n"
            "  width/height:  default 300 300\n", argv[0]);
        return 1;
    }

    const char *list_path = argv[1];
    const char *out_dir   = (argc >= 3) ? argv[2] : "batch_out";
    int W = (argc >= 5) ? atoi(argv[3]) : 300;
    int H = (argc >= 5) ? atoi(argv[4]) : 300;
    if (W <= 0) W = 300;
    if (H <= 0) H = 300;

    /* GPU init — one time */
    fprintf(stderr, "Initializing GPU...\n");
    GPU gpu = gpu_init();
    if (!gpu.device) { fprintf(stderr, "GPU init failed\n"); return 1; }

    SigilContext *ctx = sigil_create(gpu.device, WGPUTextureFormat_RGBA8Unorm, 0);
    if (!ctx) { fprintf(stderr, "sigil_create failed\n"); return 1; }
    fprintf(stderr, "GPU ready. Rendering at %dx%d.\n", W, H);

    /* Install crash handler */
    struct sigaction sa = { .sa_handler = crash_handler };
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGABRT, &sa, NULL);
    sigaction(SIGBUS,  &sa, NULL);

    FILE *list = fopen(list_path, "r");
    if (!list) { fprintf(stderr, "Cannot open %s\n", list_path); return 1; }

    int total = 0, ok = 0, parse_fail = 0, render_fail = 0;
    char line[2048];

    while (fgets(line, sizeof(line), list)) {
        line[strcspn(line, "\r\n")] = '\0';
        if (line[0] == '\0' || line[0] == '#') continue;

        total++;

        /* Load SVG */
        size_t svg_len;
        char *svg_data = read_file(line, &svg_len);
        if (!svg_data) {
            printf("NO_FILE\t%s\n", line);
            parse_fail++;
            continue;
        }

        /* Render (with crash recovery) */
        unsigned char *pixels = NULL;
        g_in_render = 1;
        if (sigsetjmp(g_jmpbuf, 1) == 0) {
            pixels = render_one(&gpu, ctx, svg_data, svg_len, W, H);
        } else {
            /* Recovered from crash */
            printf("CRASH\t%s\n", line);
            fflush(stdout);
            free(svg_data);
            render_fail++;
            continue;
        }
        g_in_render = 0;
        free(svg_data);

        if (!pixels) {
            printf("PARSE_FAIL\t%s\n", line);
            fflush(stdout);
            parse_fail++;
            continue;
        }

        /* Write PNG */
        char stem[256], outpath[512];
        get_stem(line, stem, sizeof(stem));
        snprintf(outpath, sizeof(outpath), "%s/%s.png", out_dir, stem);

        if (!stbi_write_png(outpath, W, H, 4, pixels, W * 4)) {
            printf("WRITE_FAIL\t%s\n", line);
            render_fail++;
        } else {
            printf("OK\t%s\t%s\n", line, outpath);
            ok++;
        }
        free(pixels);

        if (total % 100 == 0)
            fprintf(stderr, "[%d] ok=%d parse_fail=%d render_fail=%d\n",
                    total, ok, parse_fail, render_fail);
    }

    fclose(list);

    fprintf(stderr, "\n=== Done ===\n");
    fprintf(stderr, "Total: %d  OK: %d  Parse fail: %d  Render fail: %d\n",
            total, ok, parse_fail, render_fail);

    sigil_destroy(ctx);
    return 0;
}
