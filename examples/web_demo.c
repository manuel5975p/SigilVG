/*
 * SigilVG web demo — Emscripten build.
 * Renders SVG on a browser canvas via WebGPU.
 * SVG data is received from JavaScript (drag-and-drop).
 */

#define SIGIL_IMPLEMENTATION
#include "sigilvg.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <emscripten/emscripten.h>
#include <emscripten/html5.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef STRVIEW
#define STRVIEW(X) (WGPUStringView){X, sizeof(X) - 1}
#endif

/* ------------------------------------------------------------------ */
/*  Default SVG so the canvas isn't blank on load                     */
/* ------------------------------------------------------------------ */

static const char *DEFAULT_SVG =
    "<svg viewBox=\"0 0 400 300\">"
    "  <rect width=\"400\" height=\"300\" fill=\"#1a1a2e\"/>"
    "  <circle cx=\"200\" cy=\"130\" r=\"80\" fill=\"#e94560\"/>"
    "  <rect x=\"80\" y=\"180\" width=\"240\" height=\"60\" rx=\"12\" fill=\"#0f3460\"/>"
    "  <text x=\"200\" y=\"218\" text-anchor=\"middle\" font-size=\"24\" fill=\"white\""
    "        font-family=\"sans-serif\">Drop SVG here</text>"
    "</svg>";

/* ------------------------------------------------------------------ */
/*  Application state                                                 */
/* ------------------------------------------------------------------ */

typedef struct {
    WGPUInstance    instance;
    WGPUAdapter     adapter;
    WGPUDevice      device;
    WGPUSurface     surface;
    WGPUQueue       queue;
    GLFWwindow     *window;
    SigilContext   *sigil;
    SigilScene    *scene;
    SigilGPUScene *gpuScene;
    SigilDrawData *drawData;
    int            width;
    int            height;
    /* Camera */
    float          cam_x;
    float          cam_y;
    float          zoom;
    /* Pan state */
    int            panning;
    double         pan_last_x;
    double         pan_last_y;
} AppCtx;

static AppCtx g_app;

/* ------------------------------------------------------------------ */
/*  Camera / UBO helpers                                              */
/* ------------------------------------------------------------------ */

static void update_ubo(AppCtx *app)
{
    if (!app->drawData) return;
    float vw = (float)app->width, vh = (float)app->height;
    float z = app->zoom;
    float ubo[20] = {
        2.0f * z / vw, 0, 0, -app->cam_x * 2.0f * z / vw,
        0, -2.0f * z / vh, 0,  app->cam_y * 2.0f * z / vh,
        0, 0, 0, 0,
        0, 0, 0, 1,
        vw, vh, 0, 0,
    };
    wgpuQueueWriteBuffer(app->queue, app->drawData->uniformBuffer,
                         0, ubo, sizeof ubo);
}

static void reset_camera(AppCtx *app)
{
    app->cam_x = (float)app->width  * 0.5f;
    app->cam_y = (float)app->height * 0.5f;
    app->zoom  = 1.0f;
}

static void rebuild(AppCtx *app)
{
    if (app->drawData) {
        sigil_free_draw_data(app->drawData);
        app->drawData = NULL;
    }
    if (app->gpuScene) {
        sigil_free_gpu_scene(app->gpuScene);
        app->gpuScene = NULL;
    }
    app->gpuScene = sigil_upload(app->sigil, app->scene);
    if (!app->gpuScene) return;

    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(app->device, NULL);
    app->drawData = sigil_prepare_gpu(app->sigil, app->gpuScene, enc,
                                      (float)app->width, (float)app->height);
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
    wgpuQueueSubmit(app->queue, 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc);

    reset_camera(app);
    update_ubo(app);
}

/* ------------------------------------------------------------------ */
/*  Load new SVG (called from rebuild or JS)                          */
/* ------------------------------------------------------------------ */

static int load_svg(AppCtx *app, const char *svg, size_t len)
{
    SigilScene *scene = sigil_parse_svg(svg, len);
    if (!scene || scene->element_count == 0) {
        fprintf(stderr, "web_demo: no renderable elements in SVG\n");
        if (scene) sigil_free_scene(scene);
        return -1;
    }
    /* Replace current scene */
    if (app->drawData) { sigil_free_draw_data(app->drawData); app->drawData = NULL; }
    if (app->gpuScene) { sigil_free_gpu_scene(app->gpuScene); app->gpuScene = NULL; }
    if (app->scene) sigil_free_scene(app->scene);
    app->scene = scene;
    rebuild(app);
    fprintf(stderr, "web_demo: loaded %d element(s)\n", scene->element_count);
    return 0;
}

/* Exported to JS via EMSCRIPTEN_KEEPALIVE */
EMSCRIPTEN_KEEPALIVE
int web_demo_load_svg(const char *svg, int len)
{
    return load_svg(&g_app, svg, (size_t)len);
}

/* ------------------------------------------------------------------ */
/*  GLFW callbacks                                                    */
/* ------------------------------------------------------------------ */

static void scroll_cb(GLFWwindow *w, double xoff, double yoff)
{
    (void)xoff;
    AppCtx *app = &g_app;
    if (!app->drawData) return;

    double mx, my;
    glfwGetCursorPos(w, &mx, &my);
    int ww, wh;
    glfwGetWindowSize(w, &ww, &wh);

    float old_zoom = app->zoom;
    float factor = (yoff > 0) ? 1.1f : 1.0f / 1.1f;
    float new_zoom = old_zoom * factor;
    if (new_zoom < 0.01f) new_zoom = 0.01f;
    if (new_zoom > 1000.0f) new_zoom = 1000.0f;

    float sx = (float)mx, sy = (float)my;
    float vw = (float)ww, vh = (float)wh;
    app->cam_x += (sx - vw * 0.5f) * (1.0f / old_zoom - 1.0f / new_zoom);
    app->cam_y += (sy - vh * 0.5f) * (1.0f / old_zoom - 1.0f / new_zoom);
    app->zoom = new_zoom;
    update_ubo(app);
}

static void mouse_button_cb(GLFWwindow *w, int button, int action, int mods)
{
    (void)mods;
    AppCtx *app = &g_app;
    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        if (action == GLFW_PRESS) {
            app->panning = 1;
            glfwGetCursorPos(w, &app->pan_last_x, &app->pan_last_y);
        } else {
            app->panning = 0;
        }
    }
}

static void cursor_pos_cb(GLFWwindow *w, double xpos, double ypos)
{
    (void)w;
    AppCtx *app = &g_app;
    if (!app->panning) return;
    float dx = (float)(xpos - app->pan_last_x);
    float dy = (float)(ypos - app->pan_last_y);
    app->pan_last_x = xpos;
    app->pan_last_y = ypos;
    app->cam_x -= dx / app->zoom;
    app->cam_y -= dy / app->zoom;
    update_ubo(app);
}

static void key_cb(GLFWwindow *w, int key, int scancode, int action, int mods)
{
    (void)scancode; (void)mods;
    if (action != GLFW_PRESS) return;
    if (key == GLFW_KEY_R) {
        reset_camera(&g_app);
        update_ubo(&g_app);
    }
}

/* ------------------------------------------------------------------ */
/*  Main loop (called by emscripten_set_main_loop)                    */
/* ------------------------------------------------------------------ */

static void frame(void)
{
    AppCtx *app = &g_app;

    glfwPollEvents();

    int w, h;
    glfwGetWindowSize(app->window, &w, &h);
    if (w <= 0 || h <= 0) return;

    /* Handle resize */
    if (w != app->width || h != app->height) {
        app->width  = w;
        app->height = h;
        wgpuSurfaceConfigure(app->surface, &(const WGPUSurfaceConfiguration){
            .device      = app->device,
            .format      = WGPUTextureFormat_BGRA8Unorm,
            .usage       = WGPUTextureUsage_RenderAttachment,
            .width       = (uint32_t)w,
            .height      = (uint32_t)h,
            .alphaMode   = WGPUCompositeAlphaMode_Opaque,
            .presentMode = WGPUPresentMode_Fifo,
        });
        if (app->scene) rebuild(app);
        return;
    }

    if (!app->drawData) return;

    WGPUSurfaceTexture st;
    wgpuSurfaceGetCurrentTexture(app->surface, &st);
    if (st.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal)
        return;

    WGPUTextureView sv = wgpuTextureCreateView(st.texture,
        &(const WGPUTextureViewDescriptor){
            .format         = WGPUTextureFormat_BGRA8Unorm,
            .dimension      = WGPUTextureViewDimension_2D,
            .baseMipLevel   = 0, .mipLevelCount   = 1,
            .baseArrayLayer = 0, .arrayLayerCount  = 1,
            .aspect         = WGPUTextureAspect_All,
            .usage          = WGPUTextureUsage_RenderAttachment,
        });

    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(app->device, NULL);
    static const float bg[] = {0.102f, 0.102f, 0.180f, 1.0f}; /* #1a1a2e */
    sigil_encode(app->sigil, app->drawData, enc, sv, NULL, bg);

    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
    wgpuQueueSubmit(app->queue, 1, &cb);

    wgpuTextureViewRelease(sv);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc);
}

/* ------------------------------------------------------------------ */
/*  Main                                                              */
/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/*  Adapter / device callbacks                                        */
/* ------------------------------------------------------------------ */

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

int main(void)
{
    AppCtx *app = &g_app;
    memset(app, 0, sizeof *app);

    /* --- WebGPU instance --- */
    WGPUInstanceFeatureName ifeatures[] = {
        WGPUInstanceFeatureName_TimedWaitAny,
    };
    WGPUInstanceDescriptor idesc = {
        .requiredFeatures     = ifeatures,
        .requiredFeatureCount = 1,
    };
    app->instance = wgpuCreateInstance(&idesc);
    if (!app->instance) {
        fprintf(stderr, "Failed to create WebGPU instance\n");
        return 1;
    }

    /* --- Adapter --- */
    WGPURequestAdapterOptions aopts = {
        .featureLevel = WGPUFeatureLevel_Compatibility,
    };
    WGPURequestAdapterCallbackInfo acb = {
        .callback  = on_adapter,
        .mode      = WGPUCallbackMode_WaitAnyOnly,
        .userdata1 = &app->adapter,
    };
    WGPUFuture af = wgpuInstanceRequestAdapter(app->instance, &aopts, acb);
    wgpuInstanceWaitAny(app->instance, 1,
        &(WGPUFutureWaitInfo){.future = af}, 1000000000ULL);
    if (!app->adapter) {
        fprintf(stderr, "Failed to get WebGPU adapter\n");
        return 1;
    }

    /* --- Device --- */
    WGPUDeviceDescriptor ddesc = {0};
    WGPURequestDeviceCallbackInfo dcb = {
        .callback  = on_device,
        .mode      = WGPUCallbackMode_WaitAnyOnly,
        .userdata1 = &app->device,
    };
    WGPUFuture df = wgpuAdapterRequestDevice(app->adapter, &ddesc, dcb);
    wgpuInstanceWaitAny(app->instance, 1,
        &(WGPUFutureWaitInfo){.future = df}, 1000000000ULL);
    if (!app->device) {
        fprintf(stderr, "Failed to get WebGPU device\n");
        return 1;
    }
    app->queue = wgpuDeviceGetQueue(app->device);

    /* --- GLFW + surface --- */
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    app->window = glfwCreateWindow(900, 900, "SigilVG", NULL, NULL);

    WGPUEmscriptenSurfaceSourceCanvasHTMLSelector canvasSrc = {
        .chain = {.sType = WGPUSType_EmscriptenSurfaceSourceCanvasHTMLSelector},
        .selector = STRVIEW("canvas"),
    };
    app->surface = wgpuInstanceCreateSurface(app->instance,
        &(WGPUSurfaceDescriptor){.nextInChain = &canvasSrc.chain});

    glfwGetWindowSize(app->window, &app->width, &app->height);

    wgpuSurfaceConfigure(app->surface, &(const WGPUSurfaceConfiguration){
        .device      = app->device,
        .format      = WGPUTextureFormat_BGRA8Unorm,
        .usage       = WGPUTextureUsage_RenderAttachment,
        .width       = (uint32_t)app->width,
        .height      = (uint32_t)app->height,
        .alphaMode   = WGPUCompositeAlphaMode_Opaque,
        .presentMode = WGPUPresentMode_Fifo,
    });

    glfwSetScrollCallback(app->window, scroll_cb);
    glfwSetMouseButtonCallback(app->window, mouse_button_cb);
    glfwSetCursorPosCallback(app->window, cursor_pos_cb);
    glfwSetKeyCallback(app->window, key_cb);

    /* Create SigilVG pipeline */
    app->sigil = sigil_create(app->device, WGPUTextureFormat_BGRA8Unorm, 0);
    if (!app->sigil) {
        fprintf(stderr, "sigil_create failed\n");
        return 1;
    }

    /* Load default SVG */
    load_svg(app, DEFAULT_SVG, strlen(DEFAULT_SVG));

    fprintf(stderr, "SigilVG web demo ready\n");

    /* Emscripten main loop — never returns */
    emscripten_set_main_loop(frame, 0, 1);

    return 0;
}
