/*
 * batch_pixel_test.c — Batch pixel comparison of SigilVG against reference PNGs.
 *
 * Initializes GPU ONCE, then loops over all SVGs from a file list.
 * Compares rendered output against reference PNGs using PSNR.
 *
 * Usage: batch_pixel_test <file_list.txt>
 *   file_list.txt: one line per test, format: svg_path\tref_png_path
 *
 * Outputs CSV to stdout: svg,ref_w,ref_h,status,psnr,diff_pct
 */

#define SIGIL_IMPLEMENTATION
#include "sigilvg.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/*  GPU setup (done once)                                             */
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
    if (!g.instance) return g;

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
    if (!g.adapter) return g;

    WGPURequestDeviceCallbackInfo dcb = {
        .callback = on_device, .mode = WGPUCallbackMode_WaitAnyOnly,
        .userdata1 = &g.device,
    };
    WGPUFuture df = wgpuAdapterRequestDevice(g.adapter,
        &(WGPUDeviceDescriptor){0}, dcb);
    wgpuInstanceWaitAny(g.instance, 1, &(WGPUFutureWaitInfo){.future=df}, 2000000000);
    if (!g.device) return g;

    g.queue = wgpuDeviceGetQueue(g.device);
    return g;
}

/* ------------------------------------------------------------------ */
/*  Render one SVG                                                    */
/* ------------------------------------------------------------------ */

static volatile int g_mapped = 0;
static void map_cb(WGPUMapAsyncStatus status, WGPUStringView msg, void *u1, void *u2) {
    (void)msg; (void)u1; (void)u2;
    g_mapped = (status == WGPUMapAsyncStatus_Success) ? 1 : -1;
}

/* Returns malloc'd RGBA pixels, or NULL on failure. */
static unsigned char *render_svg(GPU *gpu, SigilContext *ctx,
                                  const char *svg_data, size_t svg_len,
                                  int w, int h)
{
    SigilScene *scene = sigil_parse_svg(svg_data, svg_len);
    if (!scene || scene->element_count == 0) { sigil_free_scene(scene); return NULL; }

    SigilDrawData *dd = sigil_prepare(ctx, scene, (float)w, (float)h, false);
    if (!dd) { sigil_free_scene(scene); return NULL; }

    WGPUTexture rt = wgpuDeviceCreateTexture(gpu->device,
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
    WGPUBuffer rbuf = wgpuDeviceCreateBuffer(gpu->device,
        &(WGPUBufferDescriptor){
            .size = (uint64_t)alignedRow * h,
            .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead,
        });

    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(gpu->device, NULL);
    static const float bg[] = {1.0f, 1.0f, 1.0f, 1.0f};
    sigil_encode(ctx, dd, enc, rtv, NULL, bg);
    wgpuCommandEncoderCopyTextureToBuffer(enc,
        &(WGPUTexelCopyTextureInfo){.texture = rt, .aspect = WGPUTextureAspect_All},
        &(WGPUTexelCopyBufferInfo){.buffer = rbuf,
            .layout = {.bytesPerRow = alignedRow, .rowsPerImage = (uint32_t)h}},
        &(WGPUExtent3D){(uint32_t)w, (uint32_t)h, 1});
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
    wgpuQueueSubmit(gpu->queue, 1, &cb);

    g_mapped = 0;
    WGPUFuture mf = wgpuBufferMapAsync(rbuf, WGPUMapMode_Read, 0,
        (size_t)alignedRow * h,
        (WGPUBufferMapCallbackInfo){.callback = map_cb, .mode = WGPUCallbackMode_WaitAnyOnly});
    wgpuInstanceWaitAny(gpu->instance, 1, &(WGPUFutureWaitInfo){.future=mf}, UINT64_MAX);

    unsigned char *pixels = NULL;
    if (g_mapped == 1) {
        const uint8_t *mapped = (const uint8_t *)wgpuBufferGetMappedRange(
            rbuf, 0, (size_t)alignedRow * h);
        pixels = (unsigned char *)malloc((size_t)w * h * 4);
        for (int y = 0; y < h; y++)
            memcpy(pixels + y * w * 4, mapped + y * alignedRow, (size_t)w * 4);
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
    sigil_free_scene(scene);
    return pixels;
}

/* ------------------------------------------------------------------ */
/*  Comparison                                                        */
/* ------------------------------------------------------------------ */

static double compute_psnr(const unsigned char *a, const unsigned char *b, int n_pixels) {
    long long sse = 0;
    int total = n_pixels * 4;
    for (int i = 0; i < total; i++) {
        int d = (int)a[i] - (int)b[i];
        sse += (long long)(d * d);
    }
    if (sse == 0) return 999.0;
    double mse = (double)sse / (double)total;
    return 10.0 * log10(255.0 * 255.0 / mse);
}

static double compute_diff_pct(const unsigned char *a, const unsigned char *b,
                                int n_pixels, int threshold) {
    int count = 0;
    for (int i = 0; i < n_pixels; i++) {
        int off = i * 4;
        int maxd = 0;
        for (int c = 0; c < 4; c++) {
            int d = abs((int)a[off+c] - (int)b[off+c]);
            if (d > maxd) maxd = d;
        }
        if (maxd > threshold) count++;
    }
    return 100.0 * (double)count / (double)n_pixels;
}

/* Composite RGBA onto white background (for reference PNGs with transparency) */
static void flatten_on_white(unsigned char *pixels, int n_pixels) {
    for (int i = 0; i < n_pixels; i++) {
        unsigned char *p = pixels + i * 4;
        int a = p[3];
        if (a == 255) continue;
        if (a == 0) { p[0] = p[1] = p[2] = 255; p[3] = 255; continue; }
        p[0] = (unsigned char)((p[0] * a + 255 * (255 - a)) / 255);
        p[1] = (unsigned char)((p[1] * a + 255 * (255 - a)) / 255);
        p[2] = (unsigned char)((p[2] * a + 255 * (255 - a)) / 255);
        p[3] = 255;
    }
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

/* ------------------------------------------------------------------ */
/*  Main                                                              */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <file_list.txt>\n", argv[0]);
        fprintf(stderr, "  file_list.txt: one line per test, tab-separated: svg_path\\tref_png_path\n");
        return 1;
    }

    /* Init GPU once */
    GPU gpu = gpu_init();
    if (!gpu.device) {
        fprintf(stderr, "GPU init failed\n");
        return 1;
    }

    SigilContext *ctx = sigil_create(gpu.device, WGPUTextureFormat_RGBA8Unorm, 0);
    if (!ctx) {
        fprintf(stderr, "sigil_create failed\n");
        return 1;
    }

    /* CSV header */
    printf("svg,ref_w,ref_h,status,psnr,diff_pct\n");
    fflush(stdout);

    FILE *list = fopen(argv[1], "r");
    if (!list) { fprintf(stderr, "Cannot open %s\n", argv[1]); return 1; }

    char line[2048];
    int total = 0, rendered = 0, good = 0, great = 0, perfect = 0;

    while (fgets(line, sizeof(line), list)) {
        /* Strip newline */
        line[strcspn(line, "\r\n")] = '\0';
        if (line[0] == '\0') continue;

        /* Parse: svg_path\tref_png_path */
        char *tab = strchr(line, '\t');
        if (!tab) continue;
        *tab = '\0';
        const char *svg_path = line;
        const char *ref_path = tab + 1;

        total++;

        /* Load reference PNG */
        int ref_w, ref_h, ref_ch;
        unsigned char *ref_pixels = stbi_load(ref_path, &ref_w, &ref_h, &ref_ch, 4);
        if (!ref_pixels) {
            printf("%s,%d,%d,NO_REF,0,0\n", svg_path, 0, 0);
            fflush(stdout);
            continue;
        }

        /* Flatten reference onto white */
        flatten_on_white(ref_pixels, ref_w * ref_h);

        /* Load SVG */
        size_t svg_len;
        char *svg_data = read_file(svg_path, &svg_len);
        if (!svg_data) {
            printf("%s,%d,%d,NO_SVG,0,0\n", svg_path, ref_w, ref_h);
            stbi_image_free(ref_pixels);
            fflush(stdout);
            continue;
        }

        /* Render */
        unsigned char *gpu_pixels = render_svg(&gpu, ctx, svg_data, svg_len, ref_w, ref_h);
        free(svg_data);

        if (!gpu_pixels) {
            printf("%s,%d,%d,RENDER_FAIL,0,0\n", svg_path, ref_w, ref_h);
            stbi_image_free(ref_pixels);
            fflush(stdout);
            continue;
        }

        rendered++;

        /* Compare */
        int npx = ref_w * ref_h;
        double psnr = compute_psnr(gpu_pixels, ref_pixels, npx);
        double diff_pct = compute_diff_pct(gpu_pixels, ref_pixels, npx, 10);

        if (psnr >= 20.0) good++;
        if (psnr >= 30.0) great++;
        if (psnr >= 40.0) perfect++;

        printf("%s,%d,%d,OK,%.2f,%.2f\n", svg_path, ref_w, ref_h, psnr, diff_pct);
        fflush(stdout);

        free(gpu_pixels);
        stbi_image_free(ref_pixels);

        /* Advance frame — resets WGVK's per-frame submit counter */
        wgpuDeviceTick(gpu.device);

        if (total % 100 == 0)
            fprintf(stderr, "[%d] rendered=%d good(>=20)=%d great(>=30)=%d perfect(>=40)=%d\n",
                    total, rendered, good, great, perfect);
    }

    fclose(list);

    /* Summary to stderr */
    fprintf(stderr, "\n=== RESULTS ===\n");
    fprintf(stderr, "Total: %d, Rendered: %d\n", total, rendered);
    fprintf(stderr, "PSNR >= 40 dB: %d (%.1f%%)\n", perfect, 100.0*perfect/rendered);
    fprintf(stderr, "PSNR >= 30 dB: %d (%.1f%%)\n", great, 100.0*great/rendered);
    fprintf(stderr, "PSNR >= 20 dB: %d (%.1f%%)\n", good, 100.0*good/rendered);
    fprintf(stderr, "PSNR <  20 dB: %d (%.1f%%)\n", rendered-good, 100.0*(rendered-good)/rendered);

    sigil_destroy(ctx);
    return 0;
}
