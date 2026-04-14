/*
 * SigilVG interactive viewer — renders SVG in a GLFW window via WebGPU.
 *
 * Usage: sigilvg_viewer [input.svg] [--capture out.png]
 *   If no input SVG given, renders a built-in test scene.
 *   --capture: render one frame, save to PNG, exit (debug mode).
 *
 * Controls:
 *   Scroll wheel        — zoom (toward cursor)
 *   Middle/right drag   — pan
 *   R                   — reset view
 *   Escape              — quit
 */

#define SIGIL_IMPLEMENTATION
#include "sigilvg.h"

#include "common.h"   /* wgpu_init(), wgpu_base, STRVIEW, nanoTime */

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
    "  <path d=\"M 50 150 L 100 50 L 150 150 Z\" fill=\"#ff4444\"/>"
    "</svg>";

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

/* Try to load a complex default SVG for benchmarking when no path is given.
   Returns NULL if none of the candidate paths exist. */
static char *try_load_default_svg(size_t *out_size, const char **out_path)
{
    static const char *candidates[] = {
        "_deps/plutosvg-src/examples/camera.svg",       /* run from build/ */
        "build/_deps/plutosvg-src/examples/camera.svg", /* run from project root */
        "../build/_deps/plutosvg-src/examples/camera.svg",
        NULL,
    };
    for (int i = 0; candidates[i]; i++) {
        FILE *f = fopen(candidates[i], "rb");
        if (!f) continue;
        fclose(f);
        char *buf = read_file(candidates[i], out_size);
        if (buf) { *out_path = candidates[i]; return buf; }
    }
    return NULL;
}

/* ------------------------------------------------------------------ */
/*  Application context                                               */
/* ------------------------------------------------------------------ */

typedef struct {
    wgpu_base     base;
    SigilContext  *sigil;
    SigilScene   *scene;
    SigilGPUScene *gpuScene;
    SigilDrawData *drawData;
    int            lastWidth;
    int            lastHeight;
    /* Camera: cam is the pixel-space point at screen center, zoom is scale */
    float          cam_x;
    float          cam_y;
    float          zoom;
    /* Pan state */
    int            panning;
    double         pan_last_x;
    double         pan_last_y;
} ViewerCtx;

static ViewerCtx *g_ctx = NULL;
static int g_resized = 0;

static void resize_cb(GLFWwindow *w, int width, int height)
{
    (void)w; (void)width; (void)height;
    g_resized = 1;
}

/* ------------------------------------------------------------------ */
/*  Update UBO with current camera transform                          */
/* ------------------------------------------------------------------ */

static void update_ubo(ViewerCtx *ctx)
{
    if (!ctx->drawData) return;

    int w, h;
    glfwGetWindowSize(ctx->base.window, &w, &h);
    float vw = (float)w, vh = (float)h;
    float z = ctx->zoom;

    float ubo[20] = {
        2.0f * z / vw, 0, 0, -ctx->cam_x * 2.0f * z / vw,
        0, -2.0f * z / vh, 0,  ctx->cam_y * 2.0f * z / vh,
        0, 0, 0, 0,
        0, 0, 0, 1,
        vw, vh, 0, 0,
    };
    wgpuQueueWriteBuffer(ctx->base.queue, ctx->drawData->uniformBuffer,
                         0, ubo, sizeof ubo);
}

/* ------------------------------------------------------------------ */
/*  Scroll callback — zoom toward cursor                              */
/* ------------------------------------------------------------------ */

static void scroll_cb(GLFWwindow *w, double xoff, double yoff)
{
    (void)xoff;
    ViewerCtx *ctx = g_ctx;
    if (!ctx) return;

    double mx, my;
    glfwGetCursorPos(w, &mx, &my);
    int ww, wh;
    glfwGetWindowSize(w, &ww, &wh);

    float old_zoom = ctx->zoom;
    float factor = (yoff > 0) ? 1.1f : 1.0f / 1.1f;
    float new_zoom = old_zoom * factor;
    if (new_zoom < 0.01f) new_zoom = 0.01f;
    if (new_zoom > 1000.0f) new_zoom = 1000.0f;

    /* Keep world point under cursor fixed */
    float sx = (float)mx, sy = (float)my;
    float vw = (float)ww, vh = (float)wh;
    ctx->cam_x += (sx - vw * 0.5f) * (1.0f / old_zoom - 1.0f / new_zoom);
    ctx->cam_y += (sy - vh * 0.5f) * (1.0f / old_zoom - 1.0f / new_zoom);
    ctx->zoom = new_zoom;

    update_ubo(ctx);
}

/* ------------------------------------------------------------------ */
/*  Mouse button callback — start/stop panning                        */
/* ------------------------------------------------------------------ */

static void mouse_button_cb(GLFWwindow *w, int button, int action, int mods)
{
    (void)mods;
    ViewerCtx *ctx = g_ctx;
    if (!ctx) return;

    if (button == GLFW_MOUSE_BUTTON_MIDDLE || button == GLFW_MOUSE_BUTTON_RIGHT) {
        if (action == GLFW_PRESS) {
            ctx->panning = 1;
            glfwGetCursorPos(w, &ctx->pan_last_x, &ctx->pan_last_y);
        } else {
            ctx->panning = 0;
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Cursor position callback — pan while dragging                     */
/* ------------------------------------------------------------------ */

static void cursor_pos_cb(GLFWwindow *w, double xpos, double ypos)
{
    (void)w;
    ViewerCtx *ctx = g_ctx;
    if (!ctx || !ctx->panning) return;

    float dx = (float)(xpos - ctx->pan_last_x);
    float dy = (float)(ypos - ctx->pan_last_y);
    ctx->pan_last_x = xpos;
    ctx->pan_last_y = ypos;

    ctx->cam_x -= dx / ctx->zoom;
    ctx->cam_y -= dy / ctx->zoom;

    update_ubo(ctx);
}

/* ------------------------------------------------------------------ */
/*  Key callback — Escape to quit, R to reset view                    */
/* ------------------------------------------------------------------ */

static void key_cb(GLFWwindow *w, int key, int scancode, int action, int mods)
{
    (void)scancode; (void)mods;
    if (action != GLFW_PRESS) return;

    if (key == GLFW_KEY_ESCAPE)
        glfwSetWindowShouldClose(w, GLFW_TRUE);

    if (key == GLFW_KEY_P && g_ctx) {
        int ww, wh;
        glfwGetWindowSize(w, &ww, &wh);
        fprintf(stderr, "view: --size %d %d --zoom %.4f --cam %.4f %.4f\n",
                ww, wh, g_ctx->zoom, g_ctx->cam_x, g_ctx->cam_y);
    }

    if (key == GLFW_KEY_R && g_ctx) {
        int ww, wh;
        glfwGetWindowSize(w, &ww, &wh);
        g_ctx->zoom  = 1.0f;
        g_ctx->cam_x = (float)ww * 0.5f;
        g_ctx->cam_y = (float)wh * 0.5f;
        update_ubo(g_ctx);
    }
}

/* ------------------------------------------------------------------ */
/*  Rebuild draw data on resize                                       */
/* ------------------------------------------------------------------ */

static void rebuild_draw_data(ViewerCtx *ctx, int width, int height)
{
    if (ctx->drawData) {
        sigil_free_draw_data(ctx->drawData);
        ctx->drawData = NULL;
    }
    if (!ctx->gpuScene) {
        ctx->gpuScene = sigil_upload(ctx->sigil, ctx->scene);
        if (!ctx->gpuScene) return;
    }
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(ctx->base.device, NULL);
    ctx->drawData = sigil_prepare_gpu(ctx->sigil, ctx->gpuScene, enc,
                                       (float)width, (float)height);
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
    wgpuQueueSubmit(ctx->base.queue, 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc);

    ctx->lastWidth  = width;
    ctx->lastHeight = height;
    ctx->cam_x = (float)width  * 0.5f;
    ctx->cam_y = (float)height * 0.5f;
    ctx->zoom  = 1.0f;
}

/* ------------------------------------------------------------------ */
/*  Main loop — one frame                                             */
/* ------------------------------------------------------------------ */

/* FPS tracker: logs avg / min / max / p99 of the last WINDOW frame times
   every time the window fills up. Frame time is measured wall-clock
   end-to-end (poll events → present). */
#define SIGIL_FPS_WINDOW 120
static void fps_tick(uint64_t frame_ns)
{
    static uint64_t samples[SIGIL_FPS_WINDOW];
    static int n = 0;

    samples[n++] = frame_ns;
    if (n < SIGIL_FPS_WINDOW) return;

    uint64_t sum = 0, mn = samples[0], mx = samples[0];
    for (int i = 0; i < SIGIL_FPS_WINDOW; i++) {
        sum += samples[i];
        if (samples[i] < mn) mn = samples[i];
        if (samples[i] > mx) mx = samples[i];
    }
    /* p99 via simple insertion sort on a copy (N=120, fine) */
    uint64_t sorted[SIGIL_FPS_WINDOW];
    memcpy(sorted, samples, sizeof sorted);
    for (int i = 1; i < SIGIL_FPS_WINDOW; i++) {
        uint64_t k = sorted[i]; int j = i - 1;
        while (j >= 0 && sorted[j] > k) { sorted[j+1] = sorted[j]; j--; }
        sorted[j+1] = k;
    }
    uint64_t p99 = sorted[(SIGIL_FPS_WINDOW * 99) / 100];

    double avg_ms = (double)sum / SIGIL_FPS_WINDOW / 1e6;
    double fps    = 1e9 * SIGIL_FPS_WINDOW / (double)sum;
    fprintf(stderr,
        "frame: avg %6.2f ms (%6.1f fps)  min %6.2f  max %6.2f  p99 %6.2f\n",
        avg_ms, fps, (double)mn / 1e6, (double)mx / 1e6, (double)p99 / 1e6);
    n = 0;
}

static void render_frame(ViewerCtx *ctx)
{
    WGPUDevice device = ctx->base.device;
    WGPUQueue  queue  = ctx->base.queue;

    uint64_t t0 = nanoTime();

    glfwPollEvents();

    int width, height;
    glfwGetWindowSize(ctx->base.window, &width, &height);
    if (width <= 0 || height <= 0) return;

    WGPUSurfaceTexture st;
    wgpuSurfaceGetCurrentTexture(ctx->base.surface, &st);

    if (st.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal || g_resized) {
        g_resized = 0;
        wgpuSurfaceConfigure(ctx->base.surface, &(const WGPUSurfaceConfiguration){
            .device      = device,
            .format      = WGPUTextureFormat_BGRA8Unorm,
            .usage       = WGPUTextureUsage_RenderAttachment,
            .width       = (uint32_t)width,
            .height      = (uint32_t)height,
            .alphaMode   = WGPUCompositeAlphaMode_Opaque,
            .presentMode = WGPUPresentMode_Fifo,
        });

        /* Rebuild draw data when viewport size changes */
        if (width != ctx->lastWidth || height != ctx->lastHeight)
            rebuild_draw_data(ctx, width, height);

        return;
    }

    if (!ctx->drawData) return;

    /* Surface texture view */
    WGPUTextureView sv = wgpuTextureCreateView(st.texture,
        &(const WGPUTextureViewDescriptor){
            .format         = WGPUTextureFormat_BGRA8Unorm,
            .dimension      = WGPUTextureViewDimension_2D,
            .baseMipLevel   = 0, .mipLevelCount   = 1,
            .baseArrayLayer = 0, .arrayLayerCount  = 1,
            .aspect         = WGPUTextureAspect_All,
            .usage          = WGPUTextureUsage_RenderAttachment,
        });

    /* Encode the SigilVG render pass */
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device, NULL);
    static const float bg[] = {1.0f, 1.0f, 1.0f, 1.0f};
    sigil_encode(ctx->sigil, ctx->drawData, enc, sv, NULL, bg);

    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
    wgpuQueueSubmit(queue, 1, &cb);
    wgpuSurfacePresent(ctx->base.surface);

    /* Release per-frame objects */
    wgpuTextureViewRelease(sv);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc);

    fps_tick(nanoTime() - t0);
}

/* ------------------------------------------------------------------ */
/*  Main                                                              */
/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/*  Capture mode: render one frame to an offscreen texture and PNG it. */
/*  Uses the viewer's exact UBO math (update_ubo) so we can diagnose   */
/*  any discrepancy between interactive vs headless output.            */
/* ------------------------------------------------------------------ */

static volatile int g_cap_mapped = 0;
static void cap_map_cb(WGPUMapAsyncStatus status, WGPUStringView msg,
                        void *u1, void *u2)
{
    (void)msg; (void)u1; (void)u2;
    g_cap_mapped = (status == WGPUMapAsyncStatus_Success) ? 1 : -1;
}

static int capture_frame(ViewerCtx *ctx, int W, int H, const char *outPath,
                          float zoom, float camX, float camY)
{
    WGPUDevice device = ctx->base.device;
    WGPUQueue  queue  = ctx->base.queue;

    /* Build draw data at the capture size */
    if (ctx->drawData) { sigil_free_draw_data(ctx->drawData); ctx->drawData = NULL; }
    ctx->gpuScene = sigil_upload(ctx->sigil, ctx->scene);
    WGPUCommandEncoder prepEnc = wgpuDeviceCreateCommandEncoder(device, NULL);
    ctx->drawData = sigil_prepare_gpu(ctx->sigil, ctx->gpuScene, prepEnc,
                                       (float)W, (float)H);
    WGPUCommandBuffer prepCb = wgpuCommandEncoderFinish(prepEnc, NULL);
    wgpuQueueSubmit(queue, 1, &prepCb);
    wgpuCommandBufferRelease(prepCb);
    wgpuCommandEncoderRelease(prepEnc);
    if (!ctx->drawData) { fprintf(stderr, "capture: sigil_prepare_gpu failed\n"); return 1; }

    /* Apply the viewer's default camera (cam centered, zoom=1) via update_ubo
       — same math as interactive path. Note update_ubo uses glfwGetWindowSize,
       so temporarily resize the window to (W,H) before calling. */
    glfwSetWindowSize(ctx->base.window, W, H);
    ctx->cam_x = (camX >= 0) ? camX : (float)W * 0.5f;
    ctx->cam_y = (camY >= 0) ? camY : (float)H * 0.5f;
    ctx->zoom  = zoom;
    update_ubo(ctx);

    /* Use BGRA8Unorm to match the interactive pipeline byte-for-byte */
    WGPUTextureFormat fmt = WGPUTextureFormat_BGRA8Unorm;
    WGPUTexture rtTex = wgpuDeviceCreateTexture(device,
        &(WGPUTextureDescriptor){
            .usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc,
            .dimension = WGPUTextureDimension_2D,
            .size = {(uint32_t)W, (uint32_t)H, 1},
            .format = fmt,
            .mipLevelCount = 1, .sampleCount = 1,
        });
    WGPUTextureView rtView = wgpuTextureCreateView(rtTex,
        &(WGPUTextureViewDescriptor){
            .format = fmt,
            .dimension = WGPUTextureViewDimension_2D,
            .mipLevelCount = 1, .arrayLayerCount = 1,
            .aspect = WGPUTextureAspect_All,
            .usage = WGPUTextureUsage_RenderAttachment,
        });

    uint32_t rowBytes   = (uint32_t)W * 4;
    uint32_t alignedRow = (rowBytes + 255) & ~255u;
    WGPUBuffer readBuf  = wgpuDeviceCreateBuffer(device,
        &(WGPUBufferDescriptor){
            .size  = (uint64_t)alignedRow * (uint64_t)H,
            .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead,
        });
    {
        size_t rbSize = (size_t)alignedRow * (size_t)H;
        void *z = calloc(1, rbSize);
        wgpuQueueWriteBuffer(queue, readBuf, 0, z, rbSize);
        free(z);
    }

    static const float bg[] = {1.0f, 1.0f, 1.0f, 1.0f};
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device, NULL);
    sigil_encode(ctx->sigil, ctx->drawData, enc, rtView, NULL, bg);
    wgpuCommandEncoderCopyTextureToBuffer(enc,
        &(WGPUTexelCopyTextureInfo){.texture = rtTex, .aspect = WGPUTextureAspect_All},
        &(WGPUTexelCopyBufferInfo){.buffer = readBuf,
            .layout = {.bytesPerRow = alignedRow, .rowsPerImage = (uint32_t)H}},
        &(WGPUExtent3D){(uint32_t)W, (uint32_t)H, 1});
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
    wgpuQueueSubmit(queue, 1, &cb);

    WGPUBufferMapCallbackInfo mi = { .callback = cap_map_cb, .mode = WGPUCallbackMode_WaitAnyOnly };
    WGPUFuture mf = wgpuBufferMapAsync(readBuf, WGPUMapMode_Read, 0,
                                        (size_t)alignedRow * H, mi);
    WGPUFutureWaitInfo wi = { .future = mf };
    wgpuInstanceWaitAny(ctx->base.instance, 1, &wi, UINT64_MAX);
    if (g_cap_mapped != 1) { fprintf(stderr, "capture: map failed\n"); return 1; }

    const uint8_t *mapped = (const uint8_t *)wgpuBufferGetMappedRange(readBuf, 0,
                                   (size_t)alignedRow * H);
    /* BGRA -> RGBA swap for PNG */
    uint8_t *rgba = (uint8_t *)malloc((size_t)W * (size_t)H * 4);
    for (int y = 0; y < H; y++) {
        const uint8_t *src = mapped + y * alignedRow;
        uint8_t *dst = rgba + y * W * 4;
        for (int x = 0; x < W; x++) {
            dst[x*4+0] = src[x*4+2];
            dst[x*4+1] = src[x*4+1];
            dst[x*4+2] = src[x*4+0];
            dst[x*4+3] = src[x*4+3];
        }
    }
    stbi_write_png(outPath, W, H, 4, rgba, W * 4);
    fprintf(stderr, "Captured %dx%d -> %s\n", W, H, outPath);
    free(rgba);

    wgpuBufferUnmap(readBuf);
    wgpuBufferDestroy(readBuf);
    wgpuBufferRelease(readBuf);
    wgpuTextureViewRelease(rtView);
    wgpuTextureDestroy(rtTex);
    wgpuTextureRelease(rtTex);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc);
    return 0;
}

int main(int argc, char **argv)
{
    const char *inputPath = NULL;
    const char *capturePath = NULL;
    int capW = 900, capH = 900;
    float capZoom = 1.0f;
    float capCamX = -1.0f, capCamY = -1.0f; /* -1 = center */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--capture") == 0 && i + 1 < argc) {
            capturePath = argv[++i];
        } else if (strcmp(argv[i], "--size") == 0 && i + 2 < argc) {
            capW = atoi(argv[++i]);
            capH = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--zoom") == 0 && i + 1 < argc) {
            capZoom = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--cam") == 0 && i + 2 < argc) {
            capCamX = (float)atof(argv[++i]);
            capCamY = (float)atof(argv[++i]);
        } else if (argv[i][0] != '-') {
            inputPath = argv[i];
        }
    }

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
        const char *found = NULL;
        svgBuf = try_load_default_svg(&svgLen, &found);
        if (svgBuf) {
            svgData = svgBuf;
            fprintf(stderr, "Input:  %s (%zu bytes) [default benchmark SVG]\n", found, svgLen);
        } else {
            svgData = DEFAULT_SVG;
            svgLen  = strlen(DEFAULT_SVG);
            fprintf(stderr, "Input:  <built-in test SVG>\n");
        }
    }

    /* Parse SVG */
    SigilScene *scene = sigil_parse_svg(svgData, svgLen);
    free(svgBuf);

    if (!scene || scene->element_count == 0) {
        fprintf(stderr, "Error: no renderable elements found in SVG\n");
        sigil_free_scene(scene);
        return 1;
    }
    fprintf(stderr, "Parsed: %d element(s)\n", scene->element_count);

    /* GLFW + WebGPU init (window, instance, adapter, device, surface) */
    ViewerCtx ctx = {0};
    ctx.base = wgpu_init();
    if (!ctx.base.device) {
        fprintf(stderr, "WGVK init failed\n");
        sigil_free_scene(scene);
        return 1;
    }

    glfwSetWindowSizeCallback(ctx.base.window, resize_cb);
    glfwSetScrollCallback(ctx.base.window, scroll_cb);
    glfwSetMouseButtonCallback(ctx.base.window, mouse_button_cb);
    glfwSetCursorPosCallback(ctx.base.window, cursor_pos_cb);
    glfwSetKeyCallback(ctx.base.window, key_cb);
    g_ctx = &ctx;

    /* Create SigilVG pipeline context (BGRA for windowed rendering) */
    ctx.sigil = sigil_create(ctx.base.device, WGPUTextureFormat_BGRA8Unorm, 0);
    if (!ctx.sigil) {
        fprintf(stderr, "Error: sigil_create failed (check shader files)\n");
        sigil_free_scene(scene);
        return 1;
    }

    ctx.scene = scene;

    /* Capture mode: render one frame to PNG and exit */
    if (capturePath) {
        int rc = capture_frame(&ctx, capW, capH, capturePath, capZoom, capCamX, capCamY);
        sigil_free_draw_data(ctx.drawData);
        sigil_free_gpu_scene(ctx.gpuScene);
        sigil_free_scene(ctx.scene);
        sigil_destroy(ctx.sigil);
        return rc;
    }

    /* Initial draw data */
    int initW, initH;
    glfwGetWindowSize(ctx.base.window, &initW, &initH);
    rebuild_draw_data(&ctx, initW, initH);

    fprintf(stderr, "SigilVG viewer ready — scroll to zoom, middle/right-drag to pan, R to reset, Esc to quit\n");

    /* Main loop */
    while (!glfwWindowShouldClose(ctx.base.window))
        render_frame(&ctx);

    /* Cleanup */
    sigil_free_draw_data(ctx.drawData);
    sigil_free_gpu_scene(ctx.gpuScene);
    sigil_free_scene(ctx.scene);
    sigil_destroy(ctx.sigil);

    return 0;
}
