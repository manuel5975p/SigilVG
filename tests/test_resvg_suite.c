/*
 * test_resvg_suite.c — Run SigilVG against resvg reference test suite.
 *
 * For each SVG+PNG pair in the resvg suite, renders the SVG via SigilVG
 * headless GPU and compares against the reference PNG using PSNR.
 *
 * Usage: sigilvg_resvg_suite [category/subcategory ...]
 *   e.g.: sigilvg_resvg_suite shapes/rect shapes/circle painting/fill
 *   With no args, runs all known-relevant categories.
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
#include <dirent.h>
#include <sys/stat.h>

/* ------------------------------------------------------------------ */
/*  GPU setup (same as test_reference.c)                              */
/* ------------------------------------------------------------------ */

typedef struct {
    WGPUInstance instance;
    WGPUAdapter  adapter;
    WGPUDevice   device;
    WGPUQueue    queue;
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
    if (g_gpu.device) return 1;

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

/* ------------------------------------------------------------------ */
/*  SigilVG headless render                                           */
/* ------------------------------------------------------------------ */

static volatile int g_mapped = 0;
static void map_cb(WGPUMapAsyncStatus status, WGPUStringView msg, void *u1, void *u2) {
    (void)msg; (void)u1; (void)u2;
    g_mapped = (status == WGPUMapAsyncStatus_Success) ? 1 : -1;
}

/* Render SVG string, return RGBA pixels (caller must free). Returns NULL on failure. */
static unsigned char *sigil_render(const char *svg, size_t svgLen, int w, int h) {
    SigilScene *scene = sigil_parse_svg(svg, svgLen);
    if (!scene || scene->element_count == 0) { sigil_free_scene(scene); return NULL; }

    SigilGPUScene *gpuScene = sigil_upload(g_sigil, scene);
    if (!gpuScene) { sigil_free_scene(scene); return NULL; }

    WGPUCommandEncoder prepEnc = wgpuDeviceCreateCommandEncoder(g_gpu.device, NULL);
    SigilDrawData *dd = sigil_prepare_gpu(g_sigil, gpuScene, prepEnc, (float)w, (float)h);
    WGPUCommandBuffer prepCb = wgpuCommandEncoderFinish(prepEnc, NULL);
    wgpuQueueSubmit(g_gpu.queue, 1, &prepCb);
    wgpuCommandBufferRelease(prepCb);
    wgpuCommandEncoderRelease(prepEnc);

    if (!dd) { sigil_free_gpu_scene(gpuScene); sigil_free_scene(scene); return NULL; }

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

    /* Pre-fill readback buffer */
    {
        size_t rbSize = (size_t)alignedRow * (size_t)h;
        void *z = calloc(1, rbSize);
        wgpuQueueWriteBuffer(g_gpu.queue, rbuf, 0, z, rbSize);
        free(z);
    }

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

    g_mapped = 0;
    WGPUFuture mf = wgpuBufferMapAsync(rbuf, WGPUMapMode_Read, 0,
        (size_t)alignedRow * h,
        (WGPUBufferMapCallbackInfo){.callback = map_cb, .mode = WGPUCallbackMode_WaitAnyOnly});
    wgpuInstanceWaitAny(g_gpu.instance, 1, &(WGPUFutureWaitInfo){.future=mf}, UINT64_MAX);

    unsigned char *result = NULL;
    if (g_mapped == 1) {
        const uint8_t *mapped = (const uint8_t *)wgpuBufferGetMappedRange(
            rbuf, 0, (size_t)alignedRow * h);
        result = (unsigned char *)malloc((size_t)w * h * 4);
        for (int y = 0; y < h; y++)
            memcpy(result + y * w * 4, mapped + y * alignedRow, (size_t)w * 4);
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
    return result;
}

/* ------------------------------------------------------------------ */
/*  Image comparison                                                  */
/* ------------------------------------------------------------------ */

static double image_psnr(const unsigned char *a, const unsigned char *b, int w, int h) {
    long long sse = 0;
    int total = w * h * 4;
    for (int i = 0; i < total; i++) {
        int d = (int)a[i] - (int)b[i];
        sse += (long long)(d * d);
    }
    if (sse == 0) return 999.0;
    double mse = (double)sse / (double)total;
    return 10.0 * log10(255.0 * 255.0 / mse);
}

static double diff_frac(const unsigned char *a, const unsigned char *b, int w, int h, int threshold) {
    int count = 0;
    for (int i = 0; i < w * h; i++) {
        int off = i * 4;
        int maxd = 0;
        for (int c = 0; c < 4; c++) {
            int d = abs((int)a[off+c] - (int)b[off+c]);
            if (d > maxd) maxd = d;
        }
        if (maxd > threshold) count++;
    }
    return (double)count / (double)(w * h);
}

/* Composite RGBA onto white background (matching SigilVG clear color) */
static void composite_on_white(unsigned char *pixels, int w, int h) {
    for (int i = 0; i < w * h; i++) {
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
/*  Test runner                                                       */
/* ------------------------------------------------------------------ */

/* Thresholds: resvg reference uses different rasterizer, so allow some diff */
#define MIN_PSNR 20.0
#define MAX_DIFF_FRAC 0.15
#define DIFF_THRESHOLD 30

typedef struct {
    int pass, fail, skip, error;
} Stats;

static int run_one_test(const char *svg_path, const char *png_path, const char *name,
                        int verbose, Stats *stats) {
    /* Load SVG */
    size_t svgLen;
    char *svg = read_file(svg_path, &svgLen);
    if (!svg) {
        if (verbose) printf("  %-50s [ERR: can't read SVG]\n", name);
        stats->error++;
        return 0;
    }

    /* Load reference PNG */
    int rw, rh, rc;
    unsigned char *ref = stbi_load(png_path, &rw, &rh, &rc, 4);
    if (!ref) {
        if (verbose) printf("  %-50s [ERR: can't load ref PNG]\n", name);
        free(svg);
        stats->error++;
        return 0;
    }

    /* Composite reference onto white (SigilVG uses white clear) */
    composite_on_white(ref, rw, rh);

    /* Render with SigilVG at reference dimensions */
    unsigned char *gpu = sigil_render(svg, svgLen, rw, rh);
    free(svg);

    if (!gpu) {
        if (verbose) printf("  %-50s [SKIP: render failed]\n", name);
        stbi_image_free(ref);
        stats->skip++;
        return 0;
    }

    /* Compare */
    double psnr = image_psnr(gpu, ref, rw, rh);
    double frac = diff_frac(gpu, ref, rw, rh, DIFF_THRESHOLD);

    int ok = (psnr >= MIN_PSNR && frac <= MAX_DIFF_FRAC);

    if (ok) {
        if (verbose) printf("  %-50s [PSNR=%.1f diff=%.1f%%] ok\n", name, psnr, frac * 100.0);
        stats->pass++;
    } else {
        printf("  %-50s [PSNR=%.1f diff=%.1f%%] FAIL\n", name, psnr, frac * 100.0);
        stats->fail++;
    }

    free(gpu);
    stbi_image_free(ref);
    return ok;
}

/* Run all SVG+PNG pairs in a directory */
static void run_directory(const char *dir_path, const char *prefix, int verbose, Stats *stats) {
    DIR *d = opendir(dir_path);
    if (!d) return;

    /* Collect SVG filenames */
    struct dirent *ent;
    char svg_paths[512][512];
    int count = 0;

    while ((ent = readdir(d)) != NULL && count < 512) {
        size_t len = strlen(ent->d_name);
        if (len < 5) continue;
        if (strcmp(ent->d_name + len - 4, ".svg") != 0) continue;
        snprintf(svg_paths[count], 512, "%s", ent->d_name);
        count++;
    }
    closedir(d);

    /* Sort for reproducible order */
    for (int i = 0; i < count - 1; i++)
        for (int j = i + 1; j < count; j++)
            if (strcmp(svg_paths[i], svg_paths[j]) > 0) {
                char tmp[512];
                memcpy(tmp, svg_paths[i], 512);
                memcpy(svg_paths[i], svg_paths[j], 512);
                memcpy(svg_paths[j], tmp, 512);
            }

    for (int i = 0; i < count; i++) {
        char svg_full[1024], png_full[1024], name[256];
        snprintf(svg_full, sizeof svg_full, "%s/%s", dir_path, svg_paths[i]);

        /* Derive PNG path: same name, .svg -> .png */
        size_t slen = strlen(svg_paths[i]);
        char png_name[512];
        memcpy(png_name, svg_paths[i], slen - 4);
        memcpy(png_name + slen - 4, ".png", 5);
        snprintf(png_full, sizeof png_full, "%s/%s", dir_path, png_name);

        /* Check PNG exists */
        struct stat st;
        if (stat(png_full, &st) != 0) {
            stats->skip++;
            continue;
        }

        /* Short display name */
        char base[256];
        memcpy(base, svg_paths[i], slen - 4);
        base[slen - 4] = '\0';
        snprintf(name, sizeof name, "%s/%s", prefix, base);

        run_one_test(svg_full, png_full, name, verbose, stats);
    }
}

/* Run a category (e.g. "shapes/rect") */
static void run_category(const char *base_dir, const char *category, int verbose, Stats *stats) {
    char dir_path[1024];
    snprintf(dir_path, sizeof dir_path, "%s/%s", base_dir, category);

    struct stat st;
    if (stat(dir_path, &st) != 0 || !S_ISDIR(st.st_mode)) {
        fprintf(stderr, "Warning: %s not found\n", dir_path);
        return;
    }

    printf("\n-- %s --\n", category);
    run_directory(dir_path, category, verbose, stats);
}

/* ------------------------------------------------------------------ */
/*  Main                                                              */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv) {
    printf("SigilVG resvg test suite runner\n");

    if (!gpu_init()) {
        fprintf(stderr, "ERROR: GPU init failed\n");
        return 1;
    }
    printf("GPU initialized\n");

    /* Base path to resvg_suite/tests */
    const char *base = "../tests/resvg_suite/tests";

    int verbose = 1;

    /* Check for -q (quiet: only show failures) */
    int first_arg = 1;
    if (argc > 1 && strcmp(argv[1], "-q") == 0) {
        verbose = 0;
        first_arg = 2;
    }
    if (argc > 1 && strcmp(argv[1], "-v") == 0) {
        verbose = 1;
        first_arg = 2;
    }

    Stats stats = {0};

    if (argc > first_arg) {
        /* Run specified categories */
        for (int i = first_arg; i < argc; i++) {
            run_category(base, argv[i], verbose, &stats);
        }
    } else {
        /* Run all categories that SigilVG can plausibly handle */

        /* Shapes */
        run_category(base, "shapes/rect", verbose, &stats);
        run_category(base, "shapes/circle", verbose, &stats);
        run_category(base, "shapes/ellipse", verbose, &stats);
        run_category(base, "shapes/line", verbose, &stats);
        run_category(base, "shapes/path", verbose, &stats);
        run_category(base, "shapes/polygon", verbose, &stats);
        run_category(base, "shapes/polyline", verbose, &stats);

        /* Painting */
        run_category(base, "painting/fill", verbose, &stats);
        run_category(base, "painting/fill-opacity", verbose, &stats);
        run_category(base, "painting/fill-rule", verbose, &stats);
        run_category(base, "painting/stroke", verbose, &stats);
        run_category(base, "painting/stroke-width", verbose, &stats);
        run_category(base, "painting/stroke-opacity", verbose, &stats);
        run_category(base, "painting/stroke-linecap", verbose, &stats);
        run_category(base, "painting/stroke-linejoin", verbose, &stats);
        run_category(base, "painting/stroke-miterlimit", verbose, &stats);
        run_category(base, "painting/stroke-dasharray", verbose, &stats);
        run_category(base, "painting/stroke-dashoffset", verbose, &stats);
        run_category(base, "painting/opacity", verbose, &stats);
        run_category(base, "painting/color", verbose, &stats);
        run_category(base, "painting/visibility", verbose, &stats);
        run_category(base, "painting/display", verbose, &stats);

        /* Structure */
        run_category(base, "structure/g", verbose, &stats);
        run_category(base, "structure/svg", verbose, &stats);
        run_category(base, "structure/transform", verbose, &stats);
        run_category(base, "structure/use", verbose, &stats);
        run_category(base, "structure/defs", verbose, &stats);
        run_category(base, "structure/symbol", verbose, &stats);

        /* Paint servers */
        run_category(base, "paint-servers/linearGradient", verbose, &stats);
        run_category(base, "paint-servers/radialGradient", verbose, &stats);
        run_category(base, "paint-servers/stop", verbose, &stats);
        run_category(base, "paint-servers/stop-color", verbose, &stats);
        run_category(base, "paint-servers/stop-opacity", verbose, &stats);
    }

    printf("\n========================================\n");
    printf("Results: %d pass, %d fail, %d skip, %d error\n",
           stats.pass, stats.fail, stats.skip, stats.error);
    printf("========================================\n");

    if (g_sigil) sigil_destroy(g_sigil);
    return 0;
}
