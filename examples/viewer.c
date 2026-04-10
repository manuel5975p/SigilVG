/*
 * SigilVG interactive viewer — renders SVG in a GLFW window via WebGPU.
 *
 * Usage: sigilvg_viewer [input.svg]
 *   If no input SVG given, renders a built-in test scene.
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

static void render_frame(ViewerCtx *ctx)
{
    WGPUDevice device = ctx->base.device;
    WGPUQueue  queue  = ctx->base.queue;

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
}

/* ------------------------------------------------------------------ */
/*  Main                                                              */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    const char *inputPath = argc >= 2 ? argv[1] : NULL;

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
        svgLen  = strlen(DEFAULT_SVG);
        fprintf(stderr, "Input:  <built-in test SVG>\n");
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
