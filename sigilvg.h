/*
 * sigilvg.h — GPU-accelerated SVG rendering via WGVK (WebGPU on Vulkan)
 *
 * Single-header library. Define SIGIL_IMPLEMENTATION in exactly one .c file
 * before including this header to get the implementation.
 *
 * Renders SVG path elements using the Slug algorithm: quadratic Bezier curves
 * evaluated analytically in fragment shaders for resolution-independent,
 * antialiased vector graphics.
 *
 * Dependencies:
 *   - WGVK (WebGPU API)
 *   - simple_wgsl (WGSL -> SPIR-V, used by WGVK)
 *   - stb_truetype (for <text> elements, bundled behind SIGIL_IMPLEMENTATION)
 */
#ifndef SIGILVG_H
#define SIGILVG_H

#include <webgpu/webgpu.h>
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  Public types                                                      */
/* ------------------------------------------------------------------ */

typedef enum {
    SIGIL_FILL_NONZERO = 0,
    SIGIL_FILL_EVENODD = 1,
} SigilFillRule;

typedef struct SigilContext  SigilContext;
typedef struct SigilScene    SigilScene;
typedef struct SigilDrawData SigilDrawData;
typedef struct SigilGPUScene SigilGPUScene;

/* ------------------------------------------------------------------ */
/*  Public API                                                        */
/* ------------------------------------------------------------------ */

SigilContext* sigil_create(WGPUDevice device, WGPUTextureFormat colorFormat,
                           WGPUTextureFormat depthFormat);

SigilScene* sigil_parse_svg(const char* svg_data, size_t len);

/* If clear_color is non-NULL, the render pass clears to that color.
   If NULL, LoadOp_Load is used (caller must have already cleared). */
void sigil_encode(SigilContext* ctx, SigilDrawData* data,
                  WGPUCommandEncoder encoder,
                  WGPUTextureView color_target,
                  WGPUTextureView depth_target,
                  const float clear_color[4]);

void sigil_free_draw_data(SigilDrawData* data);
void sigil_free_scene(SigilScene* scene);
void sigil_destroy(SigilContext* ctx);

/* GPU compute path */
SigilGPUScene* sigil_upload(SigilContext* ctx, SigilScene* scene);
void           sigil_free_gpu_scene(SigilGPUScene* gpu_scene);

SigilDrawData* sigil_prepare_gpu(SigilContext* ctx, SigilGPUScene* gpu_scene,
                                 WGPUCommandEncoder encoder,
                                 float viewport_w, float viewport_h);

/* Load a font for <text> rendering. Font data must remain valid until
   sigil_free_scene(). family_name is matched against font-family in SVG.
   Returns 0 on success, non-zero on failure. */
int sigil_load_font(SigilScene *scene, const char *family_name,
                    const unsigned char *font_data, size_t font_size);

#ifdef __cplusplus
}
#endif

/* ================================================================== */
/*  IMPLEMENTATION                                                    */
/* ================================================================== */

#ifdef SIGIL_IMPLEMENTATION

#ifdef __cplusplus
extern "C" {
/* Descriptors are passed by pointer to WGPU create-calls that copy immediately,
 * so the C99 compound-literal idiom `&(T){...}` is safe even though its
 * temporary dies at end-of-expression in C++. Likewise, designated
 * initializers out of order were legal in C99 but not C++20. */
#  if defined(__clang__)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Waddress-of-temporary"
#    pragma clang diagnostic ignored "-Wc99-designator"
#    pragma clang diagnostic ignored "-Wreorder-init-list"
#    pragma clang diagnostic ignored "-Winitializer-overrides"
#    pragma clang diagnostic ignored "-Wmissing-designated-field-initializers"
#  elif defined(__GNUC__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wpedantic"
#    pragma GCC diagnostic ignored "-Wnarrowing"
#    pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#  endif
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <ctype.h>

#ifndef STB_TRUETYPE_IMPLEMENTATION
#define STB_TRUETYPE_IMPLEMENTATION
#endif
#include "stb_truetype.h"

/* ------------------------------------------------------------------ */
/*  Constants                                                         */
/* ------------------------------------------------------------------ */

#define SIGIL_TEX_WIDTH  4096
#define SIGIL_BAND_COUNT 8

/* Forward declarations for functions used early */
static void sigil__mat_identity(float m[6]);
static float sigil__parse_length(const char *val, int vlen, float ref_size);

/* ------------------------------------------------------------------ */
/*  Internal types                                                    */
/* ------------------------------------------------------------------ */

typedef struct {
    float p0x, p0y, p1x, p1y, p2x, p2y;
} SigilCurve;

typedef struct {
    float xMin, yMin, xMax, yMax;
} SigilBounds;

typedef struct {
    int *curveIndices;
    int count;
} SigilBandEntry;

typedef struct {
    SigilBandEntry hBands[SIGIL_BAND_COUNT];
    SigilBandEntry vBands[SIGIL_BAND_COUNT];
} SigilBandData;

/* ------------------------------------------------------------------ */
/*  Gradient definitions                                              */
/* ------------------------------------------------------------------ */

#define SIGIL_GRADIENT_RAMP_WIDTH 256

typedef struct {
    float offset;
    float color[4]; /* RGBA, 0-1 */
} SigilGradientStop;

typedef struct {
    char id[128];
    int type;         /* 1 = linearGradient, 2 = radialGradient */
    SigilGradientStop *stops;
    int stop_count;
    float x1, y1, x2, y2;        /* linear: start/end; radial unused */
    float cx, cy, r, fx, fy, fr;  /* radial: center, radius, focal */
    float transform[6];
    int objectBBox;               /* 1 = objectBoundingBox, 0 = userSpaceOnUse */
    int spread;                   /* 0=pad, 1=reflect, 2=repeat */
    char href[128];               /* xlink:href for attribute inheritance */
    uint32_t attrs_set;           /* bitmask of explicitly set attributes */
} SigilGradientDef;

typedef struct {
    SigilGradientDef *data;
    int count;
    int cap;
} SigilGradientArray;

static void sigil__grad_array_init(SigilGradientArray *a) {
    a->data = NULL; a->count = 0; a->cap = 0;
}

static SigilGradientDef* sigil__grad_array_push(SigilGradientArray *a) {
    if (a->count >= a->cap) {
        a->cap = a->cap ? a->cap * 2 : 16;
        a->data = (SigilGradientDef *)realloc(a->data, (size_t)a->cap * sizeof(SigilGradientDef));
    }
    SigilGradientDef *g = &a->data[a->count++];
    memset(g, 0, sizeof(*g));
    sigil__mat_identity(g->transform);
    g->objectBBox = 1; /* SVG default: objectBoundingBox */
    /* linearGradient defaults: x1=0%, y1=0%, x2=100%, y2=0% */
    g->x2 = 1.0f;
    /* radialGradient defaults: cx=50%, cy=50%, r=50% */
    g->cx = 0.5f; g->cy = 0.5f; g->r = 0.5f;
    g->fx = -1.0f; g->fy = -1.0f; /* sentinel: fx/fy default to cx/cy */
    return g;
}

/* Check if a gradient attribute was explicitly set during parsing.
   We use sentinel values to detect unset attributes:
   - linear: x1/y1 defaults are 0, x2 default is 1 — but these are also valid values,
     so we need a separate tracking mechanism.
   We store a bitmask of which attributes were explicitly set. */
#define SIGIL_GRAD_HAS_X1      (1<<0)
#define SIGIL_GRAD_HAS_Y1      (1<<1)
#define SIGIL_GRAD_HAS_X2      (1<<2)
#define SIGIL_GRAD_HAS_Y2      (1<<3)
#define SIGIL_GRAD_HAS_CX      (1<<4)
#define SIGIL_GRAD_HAS_CY      (1<<5)
#define SIGIL_GRAD_HAS_R       (1<<6)
#define SIGIL_GRAD_HAS_FX      (1<<7)
#define SIGIL_GRAD_HAS_FY      (1<<8)
#define SIGIL_GRAD_HAS_FR      (1<<9)
#define SIGIL_GRAD_HAS_UNITS   (1<<10)
#define SIGIL_GRAD_HAS_SPREAD  (1<<11)
#define SIGIL_GRAD_HAS_XFORM   (1<<12)
/* Percent flags: set when an explicit value used '%' suffix */
#define SIGIL_GRAD_PCT_X1      (1<<13)
#define SIGIL_GRAD_PCT_Y1      (1<<14)
#define SIGIL_GRAD_PCT_X2      (1<<15)
#define SIGIL_GRAD_PCT_Y2      (1<<16)
#define SIGIL_GRAD_PCT_CX      (1<<17)
#define SIGIL_GRAD_PCT_CY      (1<<18)
#define SIGIL_GRAD_PCT_R       (1<<19)
#define SIGIL_GRAD_PCT_FX      (1<<20)
#define SIGIL_GRAD_PCT_FY      (1<<21)
#define SIGIL_GRAD_PCT_FR      (1<<22)

/* Resolve xlink:href inheritance between gradients.
   Per SVG spec, xlink:href on a gradient inherits ALL unset attributes and stops. */
static void sigil__resolve_gradient_hrefs(SigilGradientArray *arr) {
    /* Multiple passes to handle chains (A href B href C) */
    for (int pass = 0; pass < 4; pass++) {
        int changed = 0;
        for (int i = 0; i < arr->count; i++) {
            SigilGradientDef *g = &arr->data[i];
            if (g->href[0] == '\0') continue;
            const char *ref = g->href;
            if (ref[0] == '#') ref++;
            for (int j = 0; j < arr->count; j++) {
                if (i == j) continue;
                if (strcmp(arr->data[j].id, ref) == 0) {
                    SigilGradientDef *src = &arr->data[j];
                    /* Inherit stops if none defined */
                    if (g->stop_count == 0 && src->stop_count > 0) {
                        g->stop_count = src->stop_count;
                        g->stops = (SigilGradientStop *)malloc((size_t)g->stop_count * sizeof(SigilGradientStop));
                        memcpy(g->stops, src->stops, (size_t)g->stop_count * sizeof(SigilGradientStop));
                        changed = 1;
                    }
                    /* Inherit attributes that weren't explicitly set */
                    if (!(g->attrs_set & SIGIL_GRAD_HAS_UNITS) && (src->attrs_set & SIGIL_GRAD_HAS_UNITS)) {
                        g->objectBBox = src->objectBBox;
                        g->attrs_set |= SIGIL_GRAD_HAS_UNITS;
                        changed = 1;
                    }
                    if (!(g->attrs_set & SIGIL_GRAD_HAS_SPREAD) && (src->attrs_set & SIGIL_GRAD_HAS_SPREAD)) {
                        g->spread = src->spread;
                        g->attrs_set |= SIGIL_GRAD_HAS_SPREAD;
                        changed = 1;
                    }
                    if (!(g->attrs_set & SIGIL_GRAD_HAS_XFORM) && (src->attrs_set & SIGIL_GRAD_HAS_XFORM)) {
                        memcpy(g->transform, src->transform, sizeof(float)*6);
                        g->attrs_set |= SIGIL_GRAD_HAS_XFORM;
                        changed = 1;
                    }
                    /* Linear gradient attributes */
                    if (!(g->attrs_set & SIGIL_GRAD_HAS_X1) && (src->attrs_set & SIGIL_GRAD_HAS_X1)) {
                        g->x1 = src->x1; g->attrs_set |= SIGIL_GRAD_HAS_X1; changed = 1;
                    }
                    if (!(g->attrs_set & SIGIL_GRAD_HAS_Y1) && (src->attrs_set & SIGIL_GRAD_HAS_Y1)) {
                        g->y1 = src->y1; g->attrs_set |= SIGIL_GRAD_HAS_Y1; changed = 1;
                    }
                    if (!(g->attrs_set & SIGIL_GRAD_HAS_X2) && (src->attrs_set & SIGIL_GRAD_HAS_X2)) {
                        g->x2 = src->x2; g->attrs_set |= SIGIL_GRAD_HAS_X2; changed = 1;
                    }
                    if (!(g->attrs_set & SIGIL_GRAD_HAS_Y2) && (src->attrs_set & SIGIL_GRAD_HAS_Y2)) {
                        g->y2 = src->y2; g->attrs_set |= SIGIL_GRAD_HAS_Y2; changed = 1;
                    }
                    /* Radial gradient attributes */
                    if (!(g->attrs_set & SIGIL_GRAD_HAS_CX) && (src->attrs_set & SIGIL_GRAD_HAS_CX)) {
                        g->cx = src->cx; g->attrs_set |= SIGIL_GRAD_HAS_CX; changed = 1;
                    }
                    if (!(g->attrs_set & SIGIL_GRAD_HAS_CY) && (src->attrs_set & SIGIL_GRAD_HAS_CY)) {
                        g->cy = src->cy; g->attrs_set |= SIGIL_GRAD_HAS_CY; changed = 1;
                    }
                    if (!(g->attrs_set & SIGIL_GRAD_HAS_R) && (src->attrs_set & SIGIL_GRAD_HAS_R)) {
                        g->r = src->r; g->attrs_set |= SIGIL_GRAD_HAS_R; changed = 1;
                    }
                    if (!(g->attrs_set & SIGIL_GRAD_HAS_FX) && (src->attrs_set & SIGIL_GRAD_HAS_FX)) {
                        g->fx = src->fx; g->attrs_set |= SIGIL_GRAD_HAS_FX; changed = 1;
                    }
                    if (!(g->attrs_set & SIGIL_GRAD_HAS_FY) && (src->attrs_set & SIGIL_GRAD_HAS_FY)) {
                        g->fy = src->fy; g->attrs_set |= SIGIL_GRAD_HAS_FY; changed = 1;
                    }
                    if (!(g->attrs_set & SIGIL_GRAD_HAS_FR) && (src->attrs_set & SIGIL_GRAD_HAS_FR)) {
                        g->fr = src->fr; g->attrs_set |= SIGIL_GRAD_HAS_FR; changed = 1;
                    }
                    break;
                }
            }
        }
        if (!changed) break;
    }
}

typedef enum {
    SIGIL_JOIN_MITER = 0, /* SVG default */
    SIGIL_JOIN_ROUND = 1,
    SIGIL_JOIN_BEVEL = 2,
} SigilLineJoin;

typedef enum {
    SIGIL_CAP_BUTT   = 0, /* SVG default */
    SIGIL_CAP_ROUND  = 1,
    SIGIL_CAP_SQUARE = 2,
} SigilLineCap;

typedef struct {
    SigilCurve *curves;
    uint32_t    curve_count;
    float       fill_color[4];
    float       stroke_color[4];
    float       stroke_width;
    float       transform[6]; /* 2D affine: a, b, c, d, tx, ty */
    SigilFillRule fill_rule;
    float       opacity;
    SigilBounds bounds;       /* after transform */
    SigilBandData bands;      /* after transform */
    int         fill_gradient_idx; /* index into scene->gradients, or -1 */
    SigilLineJoin  stroke_linejoin;
    SigilLineCap   stroke_linecap;
    float          stroke_miterlimit;
} SigilElement;

struct SigilScene {
    SigilElement *elements;
    int           element_count;
    float         viewBox[4]; /* x, y, w, h */
    bool          has_viewBox;
    float         width, height; /* from <svg> attributes */
    int           par_align;      /* 1-9: xMin/Mid/MaxYMin/Mid/Max, default 5 (xMidYMid) */
    int           par_meet_or_slice; /* 0=meet (default), 1=slice */
    int           par_none;       /* 1 if preserveAspectRatio="none" */
    stbtt_fontinfo *fonts;
    char          **font_names;
    int             font_count;
    SigilGradientDef *gradients;
    int               gradient_count;
};

struct SigilDrawData {
    WGPUBuffer    uniformBuffer;    /* 80-byte UBO (owned) */
    WGPUBindGroup renderBindGroup;  /* references gpu_scene buffers (owned) */
    uint32_t      indexCount;
    /* Back-references for sigil_encode (not owned, point into SigilGPUScene) */
    WGPUBuffer    vertexBuf;
    WGPUBuffer    indexBuf;
};

struct SigilContext {
    WGPUDevice          device;
    WGPUQueue           queue;
    WGPUTextureFormat   colorFormat;
    WGPUTextureFormat   depthFormat;
    /* Render pipeline */
    WGPURenderPipeline  pipeline;
    WGPUBindGroupLayout renderBGL;
    WGPUPipelineLayout  renderPipelineLayout;
    WGPUShaderModule    vertexShader;
    WGPUShaderModule    fragmentShader;
    /* Compute pipelines */
    WGPUComputePipeline preparePipeline;
    WGPUComputePipeline gradientPipeline;
    WGPUBindGroupLayout prepareInputBGL;   /* group(0): scene data (read) */
    WGPUBindGroupLayout prepareOutputBGL;  /* group(1): output buffers (rw) */
    WGPUPipelineLayout  preparePipelineLayout;
    WGPUBindGroupLayout gradientBGL;
    WGPUPipelineLayout  gradientPipelineLayout;
    WGPUShaderModule    prepareShader;
    WGPUShaderModule    gradientShader;
    /* Shared */
    WGPUSampler         gradientSampler;
};

struct SigilGPUScene {
    /* Input buffers (written once by sigil_upload) */
    WGPUBuffer curvesBuf;
    WGPUBuffer elementsBuf;
    WGPUBuffer offsetsBuf;
    WGPUBuffer gradientsBuf;       /* NULL if no gradients */
    WGPUBuffer gradientStopsBuf;   /* NULL if no gradients */
    WGPUBuffer viewportBuf;        /* 32-byte uniform, written per-prepare */

    /* Output buffers (allocated once, written by compute each prepare) */
    WGPUBuffer curveBuf;           /* packed curve data for fragment shader */
    WGPUBuffer bandBuf;            /* packed band data for fragment shader */
    WGPUBuffer vertexBuf;          /* vertex buffer (also Storage for compute write) */
    WGPUBuffer indexBuf;           /* index buffer (also Storage for compute write) */
    WGPUBuffer gradientRampBuf;    /* gradient ramp staging (copied to texture) */

    /* Gradient ramp texture (RGBA8Unorm, for linear-filtered sampling) */
    WGPUTexture     gradientTexture;
    WGPUTextureView gradientView;

    /* Compute bind groups */
    WGPUBindGroup prepareInputBG;
    WGPUBindGroup prepareOutputBG;
    WGPUBindGroup gradientBG;      /* NULL if no gradients */

    /* CPU-side metadata for dispatch sizing and prepare_gpu */
    uint32_t elementCount;
    uint32_t totalCurves;
    uint32_t gradientCount;
    float    viewBox[4];           /* x, y, w, h -- copied from SigilScene */
    bool     hasViewBox;
    int      par_align;
    int      par_meet_or_slice;
    int      par_none;

    /* CPU-side copies for CPU prepare path (compute shader fallback) */
    uint32_t *cpuElemData;         /* 22 u32/f32 per element */
    uint32_t *cpuOffsetData;       /* 2 u32 per element: curve_start, band_start */
    float    *cpuCurvesData;       /* 6 floats per curve */
    uint32_t  curveOutVec4s;       /* total vec4s in curveBuf */
    uint32_t  bandOutVec4s;        /* total vec4s in bandBuf */
    SigilGradientDef *cpuGradients; /* copy of gradient defs for CPU ramp baking */
};

/* ------------------------------------------------------------------ */
/*  Internal helpers: parsing utilities                               */
/* ------------------------------------------------------------------ */

static const char* sigil__skip_ws_comma(const char *p) {
    while (*p && (isspace((unsigned char)*p) || *p == ',')) p++;
    return p;
}

static const char* sigil__parse_float(const char *p, float *out) {
    p = sigil__skip_ws_comma(p);
    char *end;
    *out = strtof(p, &end);
    if (end == p) return NULL;
    return end;
}

static const char* sigil__parse_flag(const char *p, float *out) {
    p = sigil__skip_ws_comma(p);
    if (*p == '0' || *p == '1') {
        *out = (float)(*p - '0');
        return p + 1;
    }
    return NULL;
}

/* ------------------------------------------------------------------ */
/*  Internal helpers: geometry conversions                            */
/* ------------------------------------------------------------------ */

/* Dynamic curve array helpers */
typedef struct {
    SigilCurve *data;
    int count;
    int cap;
} SigilCurveArray;

static void sigil__curve_array_init(SigilCurveArray *a) {
    a->data = NULL; a->count = 0; a->cap = 0;
}

static void sigil__curve_array_push(SigilCurveArray *a, SigilCurve c) {
    if (a->count >= a->cap) {
        a->cap = a->cap ? a->cap * 2 : 32;
        a->data = (SigilCurve *)realloc(a->data, (size_t)a->cap * sizeof(SigilCurve));
    }
    a->data[a->count++] = c;
}

/* Convert a line segment to a quadratic curve with tiny normal offset
   (same approach as slug_wgvk: offset midpoint perpendicular by 0.05) */
static SigilCurve sigil__line_to_quad(float x0, float y0, float x1, float y1) {
    float dx = x1 - x0, dy = y1 - y0;
    float len = sqrtf(dx * dx + dy * dy);
    float nx = 0.0f, ny = 0.0f;
    if (len > 1e-12f) {
        nx = -dy / len * 0.05f;
        ny =  dx / len * 0.05f;
    }
    SigilCurve c;
    c.p0x = x0; c.p0y = y0;
    c.p1x = (x0 + x1) * 0.5f + nx;
    c.p1y = (y0 + y1) * 0.5f + ny;
    c.p2x = x1; c.p2y = y1;
    return c;
}

/* Recursive cubic-to-quadratic conversion with adaptive subdivision.
   Uses best-fit quadratic control point Q = (3*(c1+c2) - p0 - p3) / 4
   and error metric: |3*(c1-c2) + p3-p0| / 6  (max deviation from quad). */
static void sigil__cubic_to_quads(float x0, float y0, float cx1, float cy1,
                                   float cx2, float cy2, float x3, float y3,
                                   SigilCurveArray *arr) {
    /* Error of best-fit quadratic approximation */
    float ex = 3.0f * (cx1 - cx2) + x3 - x0;
    float ey = 3.0f * (cy1 - cy2) + y3 - y0;
    float err2 = (ex * ex + ey * ey) / 36.0f;

    /* 0.25 pixel tolerance => err2 <= 0.0625 */
    if (err2 <= 0.0625f) {
        float qx = (3.0f * (cx1 + cx2) - x0 - x3) * 0.25f;
        float qy = (3.0f * (cy1 + cy2) - y0 - y3) * 0.25f;
        SigilCurve c = { x0, y0, qx, qy, x3, y3 };
        sigil__curve_array_push(arr, c);
        return;
    }

    /* De Casteljau split at t=0.5 */
    float m01x = (x0 + cx1) * 0.5f,  m01y = (y0 + cy1) * 0.5f;
    float m12x = (cx1 + cx2) * 0.5f, m12y = (cy1 + cy2) * 0.5f;
    float m23x = (cx2 + x3) * 0.5f,  m23y = (cy2 + y3) * 0.5f;
    float m012x = (m01x + m12x) * 0.5f, m012y = (m01y + m12y) * 0.5f;
    float m123x = (m12x + m23x) * 0.5f, m123y = (m12y + m23y) * 0.5f;
    float midx  = (m012x + m123x) * 0.5f, midy = (m012y + m123y) * 0.5f;

    sigil__cubic_to_quads(x0, y0, m01x, m01y, m012x, m012y, midx, midy, arr);
    sigil__cubic_to_quads(midx, midy, m123x, m123y, m23x, m23y, x3, y3, arr);
}

/* SVG arc endpoint parameterization -> cubic beziers -> quadratics */
static void sigil__arc_to_cubics(float x1, float y1,
                                  float rx, float ry, float phi_deg,
                                  float fA, float fS,
                                  float x2, float y2,
                                  SigilCurveArray *arr) {
    /* SVG arc endpoint -> center parameterization, then approximate with cubics */
    if (fabsf(rx) < 1e-6f || fabsf(ry) < 1e-6f) {
        sigil__curve_array_push(arr, sigil__line_to_quad(x1, y1, x2, y2));
        return;
    }
    rx = fabsf(rx); ry = fabsf(ry);

    float phi = phi_deg * 3.14159265358979f / 180.0f;
    float cosPhi = cosf(phi), sinPhi = sinf(phi);

    /* Step 1: compute (x1', y1') */
    float dx2 = (x1 - x2) * 0.5f, dy2 = (y1 - y2) * 0.5f;
    float x1p =  cosPhi * dx2 + sinPhi * dy2;
    float y1p = -sinPhi * dx2 + cosPhi * dy2;

    /* Step 2: correct radii if needed */
    float x1p2 = x1p * x1p, y1p2 = y1p * y1p;
    float rx2 = rx * rx, ry2 = ry * ry;
    float lambda = x1p2 / rx2 + y1p2 / ry2;
    if (lambda > 1.0f) {
        float sl = sqrtf(lambda);
        rx *= sl; ry *= sl;
        rx2 = rx * rx; ry2 = ry * ry;
    }

    /* Step 3: compute center (cx', cy') */
    float num = rx2 * ry2 - rx2 * y1p2 - ry2 * x1p2;
    float den = rx2 * y1p2 + ry2 * x1p2;
    float sq = 0.0f;
    if (den > 1e-12f && num > 0.0f) sq = sqrtf(num / den);
    if ((fA != 0.0f) == (fS != 0.0f)) sq = -sq;
    float cxp =  sq * rx * y1p / ry;
    float cyp = -sq * ry * x1p / rx;

    /* Step 4: compute center (cx, cy) */
    float cx = cosPhi * cxp - sinPhi * cyp + (x1 + x2) * 0.5f;
    float cy = sinPhi * cxp + cosPhi * cyp + (y1 + y2) * 0.5f;

    /* Step 5: compute theta1, dtheta */
    float ux = (x1p - cxp) / rx, uy = (y1p - cyp) / ry;
    float vx = (-x1p - cxp) / rx, vy = (-y1p - cyp) / ry;

    float uLen = sqrtf(ux * ux + uy * uy);
    if (uLen < 1e-12f) uLen = 1e-12f;
    float theta1 = acosf(fmaxf(-1.0f, fminf(1.0f, ux / uLen)));
    if (uy < 0.0f) theta1 = -theta1;

    float dp = ux * vx + uy * vy;
    float cross = ux * vy - uy * vx;
    float uvLen = uLen * sqrtf(vx * vx + vy * vy);
    if (uvLen < 1e-12f) uvLen = 1e-12f;
    float dtheta = acosf(fmaxf(-1.0f, fminf(1.0f, dp / uvLen)));
    if (cross < 0.0f) dtheta = -dtheta;

    float PI = 3.14159265358979f;
    if (fS != 0.0f && dtheta < 0.0f) dtheta += 2.0f * PI;
    else if (fS == 0.0f && dtheta > 0.0f) dtheta -= 2.0f * PI;

    /* Split into segments of at most pi/2 */
    int nSegs = (int)ceilf(fabsf(dtheta) / (PI * 0.5f));
    if (nSegs < 1) nSegs = 1;
    float segAngle = dtheta / (float)nSegs;

    /* Cubic approximation of a circular arc of angle alpha:
       control tangent length = (4/3)*tan(alpha/4) */
    float alpha = segAngle;
    float t = (4.0f / 3.0f) * tanf(alpha * 0.25f);

    float prevX = x1, prevY = y1;
    for (int i = 0; i < nSegs; i++) {
        float a1 = theta1 + (float)i * segAngle;
        float a2 = a1 + segAngle;

        float cosA1 = cosf(a1), sinA1 = sinf(a1);
        float cosA2 = cosf(a2), sinA2 = sinf(a2);

        /* Control points in the rotated/scaled unit circle */
        float ep1x = cosA1, ep1y = sinA1;
        float ep2x = cosA2, ep2y = sinA2;

        float c1x = ep1x - t * sinA1, c1y = ep1y + t * cosA1;
        float c2x = ep2x + t * sinA2, c2y = ep2y - t * cosA2;

        /* Transform back to original coordinate system */
        float bx1 = cosPhi * rx * c1x - sinPhi * ry * c1y + cx;
        float by1 = sinPhi * rx * c1x + cosPhi * ry * c1y + cy;
        float bx2 = cosPhi * rx * c2x - sinPhi * ry * c2y + cx;
        float by2 = sinPhi * rx * c2x + cosPhi * ry * c2y + cy;
        float ex  = cosPhi * rx * ep2x - sinPhi * ry * ep2y + cx;
        float ey  = sinPhi * rx * ep2x + cosPhi * ry * ep2y + cy;

        sigil__cubic_to_quads(prevX, prevY, bx1, by1, bx2, by2, ex, ey, arr);
        prevX = ex; prevY = ey;
    }
}

/* ------------------------------------------------------------------ */
/*  SVG Path 'd' attribute parser                                     */
/* ------------------------------------------------------------------ */

static int sigil__parse_path(const char *d, SigilCurve **out_curves,
                             SigilBounds *out_bounds) {
    if (!d || !out_curves || !out_bounds) return 0;

    SigilCurveArray arr;
    sigil__curve_array_init(&arr);

    float cx = 0, cy = 0;       /* current point */
    float startX = 0, startY = 0; /* subpath start */
    float lastCx = 0, lastCy = 0; /* last control point (for S/T) */
    char lastCmd = 0;

    const char *p = d;
    char cmd = 0;

    while (*p) {
        p = sigil__skip_ws_comma(p);
        if (!*p) break;

        /* Check for a new command letter */
        if (isalpha((unsigned char)*p)) {
            cmd = *p++;
        }

        /* Some commands take no coordinates (Z/z) */
        if (cmd == 'Z' || cmd == 'z') {
            /* Close path: line from current to start */
            if (fabsf(cx - startX) > 1e-6f || fabsf(cy - startY) > 1e-6f) {
                sigil__curve_array_push(&arr, sigil__line_to_quad(cx, cy, startX, startY));
            }
            cx = startX; cy = startY;
            lastCx = cx; lastCy = cy;
            lastCmd = cmd;
            continue;
        }

        float vals[7];
        int nvals = 0;

        switch (cmd) {
        case 'M': case 'm': nvals = 2; break;
        case 'L': case 'l': nvals = 2; break;
        case 'H': case 'h': nvals = 1; break;
        case 'V': case 'v': nvals = 1; break;
        case 'C': case 'c': nvals = 6; break;
        case 'S': case 's': nvals = 4; break;
        case 'Q': case 'q': nvals = 4; break;
        case 'T': case 't': nvals = 2; break;
        case 'A': case 'a': nvals = 7; break;
        default: p++; continue;
        }

        /* Parse the required number of values */
        int ok = 1;
        for (int i = 0; i < nvals; i++) {
            if (cmd == 'A' || cmd == 'a') {
                /* Arc has special flag parsing for 4th and 5th params */
                if (i == 3 || i == 4) {
                    const char *np = sigil__parse_flag(p, &vals[i]);
                    if (!np) { ok = 0; break; }
                    p = np;
                } else {
                    const char *np = sigil__parse_float(p, &vals[i]);
                    if (!np) { ok = 0; break; }
                    p = np;
                }
            } else {
                const char *np = sigil__parse_float(p, &vals[i]);
                if (!np) { ok = 0; break; }
                p = np;
            }
        }
        if (!ok) { p++; continue; }

        switch (cmd) {
        case 'M':
            cx = vals[0]; cy = vals[1];
            startX = cx; startY = cy;
            lastCx = cx; lastCy = cy;
            /* After first M coord pair, subsequent pairs are treated as L */
            cmd = 'L';
            break;
        case 'm':
            cx += vals[0]; cy += vals[1];
            startX = cx; startY = cy;
            lastCx = cx; lastCy = cy;
            cmd = 'l';
            break;

        case 'L': {
            float x = vals[0], y = vals[1];
            sigil__curve_array_push(&arr, sigil__line_to_quad(cx, cy, x, y));
            cx = x; cy = y;
            lastCx = cx; lastCy = cy;
            break;
        }
        case 'l': {
            float x = cx + vals[0], y = cy + vals[1];
            sigil__curve_array_push(&arr, sigil__line_to_quad(cx, cy, x, y));
            cx = x; cy = y;
            lastCx = cx; lastCy = cy;
            break;
        }

        case 'H': {
            float x = vals[0];
            sigil__curve_array_push(&arr, sigil__line_to_quad(cx, cy, x, cy));
            cx = x;
            lastCx = cx; lastCy = cy;
            break;
        }
        case 'h': {
            float x = cx + vals[0];
            sigil__curve_array_push(&arr, sigil__line_to_quad(cx, cy, x, cy));
            cx = x;
            lastCx = cx; lastCy = cy;
            break;
        }

        case 'V': {
            float y = vals[0];
            sigil__curve_array_push(&arr, sigil__line_to_quad(cx, cy, cx, y));
            cy = y;
            lastCx = cx; lastCy = cy;
            break;
        }
        case 'v': {
            float y = cy + vals[0];
            sigil__curve_array_push(&arr, sigil__line_to_quad(cx, cy, cx, y));
            cy = y;
            lastCx = cx; lastCy = cy;
            break;
        }

        case 'C': {
            float x1 = vals[0], y1 = vals[1];
            float x2 = vals[2], y2 = vals[3];
            float x  = vals[4], y  = vals[5];
            sigil__cubic_to_quads(cx, cy, x1, y1, x2, y2, x, y, &arr);
            lastCx = x2; lastCy = y2;
            cx = x; cy = y;
            break;
        }
        case 'c': {
            float x1 = cx + vals[0], y1 = cy + vals[1];
            float x2 = cx + vals[2], y2 = cy + vals[3];
            float x  = cx + vals[4], y  = cy + vals[5];
            sigil__cubic_to_quads(cx, cy, x1, y1, x2, y2, x, y, &arr);
            lastCx = x2; lastCy = y2;
            cx = x; cy = y;
            break;
        }

        case 'S': {
            /* Reflected control point from previous cubic */
            float rx = cx, ry = cy;
            if (lastCmd == 'C' || lastCmd == 'c' || lastCmd == 'S' || lastCmd == 's') {
                rx = 2.0f * cx - lastCx;
                ry = 2.0f * cy - lastCy;
            }
            float x2 = vals[0], y2 = vals[1];
            float x  = vals[2], y  = vals[3];
            sigil__cubic_to_quads(cx, cy, rx, ry, x2, y2, x, y, &arr);
            lastCx = x2; lastCy = y2;
            cx = x; cy = y;
            break;
        }
        case 's': {
            float rx = cx, ry = cy;
            if (lastCmd == 'C' || lastCmd == 'c' || lastCmd == 'S' || lastCmd == 's') {
                rx = 2.0f * cx - lastCx;
                ry = 2.0f * cy - lastCy;
            }
            float x2 = cx + vals[0], y2 = cy + vals[1];
            float x  = cx + vals[2], y  = cy + vals[3];
            sigil__cubic_to_quads(cx, cy, rx, ry, x2, y2, x, y, &arr);
            lastCx = x2; lastCy = y2;
            cx = x; cy = y;
            break;
        }

        case 'Q': {
            float qx = vals[0], qy = vals[1];
            float x  = vals[2], y  = vals[3];
            SigilCurve c = { cx, cy, qx, qy, x, y };
            sigil__curve_array_push(&arr, c);
            lastCx = qx; lastCy = qy;
            cx = x; cy = y;
            break;
        }
        case 'q': {
            float qx = cx + vals[0], qy = cy + vals[1];
            float x  = cx + vals[2], y  = cy + vals[3];
            SigilCurve c = { cx, cy, qx, qy, x, y };
            sigil__curve_array_push(&arr, c);
            lastCx = qx; lastCy = qy;
            cx = x; cy = y;
            break;
        }

        case 'T': {
            float qx = cx, qy = cy;
            if (lastCmd == 'Q' || lastCmd == 'q' || lastCmd == 'T' || lastCmd == 't') {
                qx = 2.0f * cx - lastCx;
                qy = 2.0f * cy - lastCy;
            }
            float x = vals[0], y = vals[1];
            SigilCurve c = { cx, cy, qx, qy, x, y };
            sigil__curve_array_push(&arr, c);
            lastCx = qx; lastCy = qy;
            cx = x; cy = y;
            break;
        }
        case 't': {
            float qx = cx, qy = cy;
            if (lastCmd == 'Q' || lastCmd == 'q' || lastCmd == 'T' || lastCmd == 't') {
                qx = 2.0f * cx - lastCx;
                qy = 2.0f * cy - lastCy;
            }
            float x = cx + vals[0], y = cy + vals[1];
            SigilCurve c = { cx, cy, qx, qy, x, y };
            sigil__curve_array_push(&arr, c);
            lastCx = qx; lastCy = qy;
            cx = x; cy = y;
            break;
        }

        case 'A': {
            float arx = vals[0], ary = vals[1], rot = vals[2];
            float fa = vals[3], fs = vals[4];
            float x = vals[5], y = vals[6];
            sigil__arc_to_cubics(cx, cy, arx, ary, rot, fa, fs, x, y, &arr);
            cx = x; cy = y;
            lastCx = cx; lastCy = cy;
            break;
        }
        case 'a': {
            float arx = vals[0], ary = vals[1], rot = vals[2];
            float fa = vals[3], fs = vals[4];
            float x = cx + vals[5], y = cy + vals[6];
            sigil__arc_to_cubics(cx, cy, arx, ary, rot, fa, fs, x, y, &arr);
            cx = x; cy = y;
            lastCx = cx; lastCy = cy;
            break;
        }

        default: break;
        }

        lastCmd = cmd;
    }

    /* Compute bounds */
    out_bounds->xMin =  FLT_MAX; out_bounds->yMin =  FLT_MAX;
    out_bounds->xMax = -FLT_MAX; out_bounds->yMax = -FLT_MAX;
    for (int i = 0; i < arr.count; i++) {
        SigilCurve *c = &arr.data[i];
        float xs[3] = { c->p0x, c->p1x, c->p2x };
        float ys[3] = { c->p0y, c->p1y, c->p2y };
        for (int j = 0; j < 3; j++) {
            if (xs[j] < out_bounds->xMin) out_bounds->xMin = xs[j];
            if (xs[j] > out_bounds->xMax) out_bounds->xMax = xs[j];
            if (ys[j] < out_bounds->yMin) out_bounds->yMin = ys[j];
            if (ys[j] > out_bounds->yMax) out_bounds->yMax = ys[j];
        }
    }
    if (arr.count == 0) {
        out_bounds->xMin = out_bounds->yMin = 0;
        out_bounds->xMax = out_bounds->yMax = 0;
    }

    *out_curves = arr.data;
    return arr.count;
}

/* ------------------------------------------------------------------ */
/*  XML parser helpers                                                */
/* ------------------------------------------------------------------ */

typedef struct {
    const char *name;      /* pointer into source, NOT null-terminated */
    int         name_len;
    const char *attrs;     /* pointer into source to start of attrs */
    int         attrs_len;
    int         self_close; /* 1 if /> */
    int         is_close;   /* 1 if </tag> */
} SigilTag;

/* Find next XML tag starting from *pos. Returns 1 if found. */
static int sigil__next_tag(const char *src, int len, int *pos, SigilTag *tag) {
    while (*pos < len) {
        if (src[*pos] == '<') {
            int start = *pos + 1;
            if (start >= len) return 0;

            /* Skip comments */
            if (start + 2 < len && src[start] == '!' && src[start+1] == '-' && src[start+2] == '-') {
                /* Find --> */
                const char *end = strstr(src + start, "-->");
                if (end) { *pos = (int)(end - src) + 3; continue; }
                else return 0;
            }
            /* Skip <? processing instructions */
            if (src[start] == '?') {
                const char *end = strstr(src + start, "?>");
                if (end) { *pos = (int)(end - src) + 2; continue; }
                else return 0;
            }
            /* Skip <!DOCTYPE etc */
            if (src[start] == '!') {
                while (*pos < len && src[*pos] != '>') (*pos)++;
                if (*pos < len) (*pos)++;
                continue;
            }

            tag->is_close = 0;
            tag->self_close = 0;

            int p = start;
            if (src[p] == '/') { tag->is_close = 1; p++; }

            /* Tag name */
            tag->name = src + p;
            while (p < len && !isspace((unsigned char)src[p]) && src[p] != '>' && src[p] != '/') p++;
            tag->name_len = (int)(src + p - tag->name);

            /* Attributes */
            while (p < len && isspace((unsigned char)src[p])) p++;
            tag->attrs = src + p;

            /* Find end of tag */
            while (p < len && src[p] != '>') {
                if (src[p] == '/' && p + 1 < len && src[p+1] == '>') {
                    tag->self_close = 1;
                    tag->attrs_len = (int)(src + p - tag->attrs);
                    *pos = p + 2;
                    return 1;
                }
                p++;
            }
            tag->attrs_len = (int)(src + p - tag->attrs);
            if (p < len) *pos = p + 1;
            else *pos = len;
            return 1;
        }
        (*pos)++;
    }
    return 0;
}

/* Extract attribute value by name. Returns length, fills out_val. */
static int sigil__get_attr(const char *attrs, int attrs_len,
                            const char *name, const char **out_val) {
    int nlen = (int)strlen(name);
    const char *p = attrs;
    const char *end = attrs + attrs_len;

    while (p < end) {
        /* Skip whitespace */
        while (p < end && isspace((unsigned char)*p)) p++;
        if (p >= end) break;

        /* Attribute name */
        const char *aname = p;
        while (p < end && *p != '=' && !isspace((unsigned char)*p)) p++;
        int alen = (int)(p - aname);

        /* Skip to = */
        while (p < end && isspace((unsigned char)*p)) p++;
        if (p >= end || *p != '=') continue;
        p++; /* skip = */
        while (p < end && isspace((unsigned char)*p)) p++;
        if (p >= end) break;

        char quote = *p;
        if (quote != '"' && quote != '\'') continue;
        p++; /* skip opening quote */
        const char *vstart = p;
        while (p < end && *p != quote) p++;
        int vlen = (int)(p - vstart);
        if (p < end) p++; /* skip closing quote */

        if (alen == nlen && memcmp(aname, name, (size_t)nlen) == 0) {
            *out_val = vstart;
            return vlen;
        }
    }
    *out_val = NULL;
    return 0;
}

/* Parse a single CSS property from an inline style string.
   e.g. style="fill:red; stroke:blue" — looks up "fill" and returns "red". */
static int sigil__get_style_prop(const char *style, int style_len,
                                  const char *name, const char **out_val) {
    if (!style || style_len <= 0) { *out_val = NULL; return 0; }
    int nlen = (int)strlen(name);
    const char *p = style;
    const char *end = style + style_len;
    while (p < end) {
        while (p < end && (isspace((unsigned char)*p) || *p == ';')) p++;
        if (p >= end) break;
        const char *pname = p;
        while (p < end && *p != ':' && *p != ';') p++;
        if (p >= end || *p != ':') { while (p < end && *p != ';') p++; continue; }
        int pnlen = (int)(p - pname);
        while (pnlen > 0 && isspace((unsigned char)pname[pnlen-1])) pnlen--;
        p++; /* skip ':' */
        while (p < end && isspace((unsigned char)*p)) p++;
        const char *vstart = p;
        while (p < end && *p != ';') p++;
        int vlen = (int)(p - vstart);
        while (vlen > 0 && isspace((unsigned char)vstart[vlen-1])) vlen--;
        if (pnlen == nlen && memcmp(pname, name, (size_t)nlen) == 0) {
            *out_val = vstart;
            return vlen;
        }
    }
    *out_val = NULL;
    return 0;
}

/* Get a property value, checking inline style first, then presentation attribute.
   SVG spec: inline style overrides presentation attributes. */
static int sigil__get_prop(const char *attrs, int attrs_len,
                            const char *style, int style_len,
                            const char *name, const char **out_val) {
    int vlen = sigil__get_style_prop(style, style_len, name, out_val);
    if (vlen > 0) return vlen;
    return sigil__get_attr(attrs, attrs_len, name, out_val);
}

static float sigil__get_prop_float(const char *attrs, int attrs_len,
                                    const char *style, int style_len,
                                    const char *name, float def) {
    const char *val;
    int vlen = sigil__get_prop(attrs, attrs_len, style, style_len, name, &val);
    if (vlen == 0 || !val) return def;
    return sigil__parse_length(val, vlen, 16.0f);
}

/* Global viewport dimensions for CSS viewport-relative units.
   Set by the parser when processing the root <svg> element. */
static float sigil__vp_width_global = 300.0f;
static float sigil__vp_height_global = 150.0f;
static float sigil__font_size_global = 16.0f; /* root font size for rem/em */

/* Parse a CSS length value with optional unit suffix.
   ref_size is used for em/% (default font-size or viewport dimension).
   DPI assumption: 96 (CSS reference pixel). */
static float sigil__parse_length(const char *val, int vlen, float ref_size) {
    if (!val || vlen <= 0) return 0.0f;
    char *end;
    float v = strtof(val, &end);
    int remaining = vlen - (int)(end - val);
    while (remaining > 0 && isspace((unsigned char)*end)) { end++; remaining--; }
    /* Helper: check no trailing alpha after N-char unit (e.g., "mmx" is invalid) */
    #define SIGIL_UNIT_OK(n) (remaining == (n) || !isalpha((unsigned char)end[n]))
    if (remaining >= 4) {
        if (end[0]=='v' && end[1]=='m' && end[2]=='i' && end[3]=='n' && SIGIL_UNIT_OK(4))
            return v * fminf(sigil__vp_width_global, sigil__vp_height_global) / 100.0f;
        if (end[0]=='v' && end[1]=='m' && end[2]=='a' && end[3]=='x' && SIGIL_UNIT_OK(4))
            return v * fmaxf(sigil__vp_width_global, sigil__vp_height_global) / 100.0f;
    }
    if (remaining >= 3) {
        if (end[0]=='r' && end[1]=='e' && end[2]=='m' && SIGIL_UNIT_OK(3))
            return v * sigil__font_size_global;
        if (end[0]=='r' && end[1]=='l' && end[2]=='h' && SIGIL_UNIT_OK(3))
            return v * sigil__font_size_global * 1.2f; /* rlh = root line-height, ~1.2em */
    }
    if (remaining >= 2) {
        if (end[0]=='p' && end[1]=='x' && SIGIL_UNIT_OK(2)) return v;
        if (end[0]=='p' && end[1]=='t' && SIGIL_UNIT_OK(2)) return v * (96.0f / 72.0f);
        if (end[0]=='p' && end[1]=='c' && SIGIL_UNIT_OK(2)) return v * 16.0f;
        if (end[0]=='m' && end[1]=='m' && SIGIL_UNIT_OK(2)) return v * (96.0f / 25.4f);
        if (end[0]=='c' && end[1]=='m' && SIGIL_UNIT_OK(2)) return v * (96.0f / 2.54f);
        if (end[0]=='i' && end[1]=='n' && SIGIL_UNIT_OK(2)) return v * 96.0f;
        if (end[0]=='e' && end[1]=='m' && SIGIL_UNIT_OK(2)) return v * ref_size;
        if (end[0]=='e' && end[1]=='x' && SIGIL_UNIT_OK(2)) return v * ref_size * 0.5f;
        if (end[0]=='c' && end[1]=='h' && SIGIL_UNIT_OK(2)) return v * ref_size * 0.5f;
        if (end[0]=='i' && end[1]=='c' && SIGIL_UNIT_OK(2)) return v * ref_size;
        if (end[0]=='l' && end[1]=='h' && SIGIL_UNIT_OK(2)) return v * ref_size * 1.2f;
        if (end[0]=='v' && end[1]=='w' && SIGIL_UNIT_OK(2)) return v * sigil__vp_width_global / 100.0f;
        if (end[0]=='v' && end[1]=='h' && SIGIL_UNIT_OK(2)) return v * sigil__vp_height_global / 100.0f;
        if (end[0]=='v' && end[1]=='i' && SIGIL_UNIT_OK(2)) return v * sigil__vp_width_global / 100.0f;
        if (end[0]=='v' && end[1]=='b' && SIGIL_UNIT_OK(2)) return v * sigil__vp_height_global / 100.0f;
    }
    if (remaining >= 1) {
        if (end[0] == '%') return v * ref_size / 100.0f;
        if (end[0] == 'q' && SIGIL_UNIT_OK(1)) return v * (96.0f / 101.6f); /* 1q = 1/4 mm */
        /* Unknown unit suffix — treat as invalid (return 0) */
        if (isalpha((unsigned char)end[0])) return 0.0f;
    }
    #undef SIGIL_UNIT_OK
    return v; /* unitless = user units (px) */
}

static float sigil__get_attr_float(const char *attrs, int attrs_len,
                                    const char *name, float def) {
    const char *val;
    int vlen = sigil__get_attr(attrs, attrs_len, name, &val);
    if (vlen == 0 || !val) return def;
    return sigil__parse_length(val, vlen, 16.0f);
}

/* Viewport-aware variant: ref_size = viewport dimension for % resolution.
   For em/ex units, parse_length uses the global font_size. The ref_size here
   is used for % resolution. */
static float sigil__get_attr_vp(const char *attrs, int alen,
                                 const char *name, float def, float vp_ref) {
    const char *val;
    int vlen = sigil__get_attr(attrs, alen, name, &val);
    if (vlen == 0 || !val) return def;
    /* Check if value contains em/ex/ch/etc — use font_size_global for those */
    char *ep;
    float v = strtof(val, &ep);
    int rem = vlen - (int)(ep - val);
    while (rem > 0 && isspace((unsigned char)*ep)) { ep++; rem--; }
    if (rem >= 2 && ((ep[0]=='e' && ep[1]=='m') || (ep[0]=='e' && ep[1]=='x') ||
                     (ep[0]=='c' && ep[1]=='h') || (ep[0]=='i' && ep[1]=='c') ||
                     (ep[0]=='l' && ep[1]=='h'))) {
        return sigil__parse_length(val, vlen, sigil__font_size_global);
    }
    if (rem >= 3 && ep[0]=='r' && ep[1]=='l' && ep[2]=='h') {
        return sigil__parse_length(val, vlen, sigil__font_size_global);
    }
    (void)v;
    return sigil__parse_length(val, vlen, vp_ref);
}

/* ------------------------------------------------------------------ */
/*  Color parsing                                                     */
/* ------------------------------------------------------------------ */

typedef struct { const char *name; float r, g, b; } SigilNamedColor;

static const SigilNamedColor sigil__named_colors[] = {
    {"aliceblue",0.941f,0.973f,1.0f},{"antiquewhite",0.980f,0.922f,0.843f},
    {"aqua",0.0f,1.0f,1.0f},{"aquamarine",0.498f,1.0f,0.831f},
    {"azure",0.941f,1.0f,1.0f},{"beige",0.961f,0.961f,0.863f},
    {"bisque",1.0f,0.894f,0.769f},{"black",0.0f,0.0f,0.0f},
    {"blanchedalmond",1.0f,0.922f,0.804f},{"blue",0.0f,0.0f,1.0f},
    {"blueviolet",0.541f,0.169f,0.886f},{"brown",0.647f,0.165f,0.165f},
    {"burlywood",0.871f,0.722f,0.529f},{"cadetblue",0.373f,0.620f,0.627f},
    {"chartreuse",0.498f,1.0f,0.0f},{"chocolate",0.824f,0.412f,0.118f},
    {"coral",1.0f,0.498f,0.314f},{"cornflowerblue",0.392f,0.584f,0.929f},
    {"cornsilk",1.0f,0.973f,0.863f},{"crimson",0.863f,0.078f,0.235f},
    {"cyan",0.0f,1.0f,1.0f},{"darkblue",0.0f,0.0f,0.545f},
    {"darkcyan",0.0f,0.545f,0.545f},{"darkgoldenrod",0.722f,0.525f,0.043f},
    {"darkgray",0.663f,0.663f,0.663f},{"darkgreen",0.0f,0.392f,0.0f},
    {"darkgrey",0.663f,0.663f,0.663f},{"darkkhaki",0.741f,0.718f,0.420f},
    {"darkmagenta",0.545f,0.0f,0.545f},{"darkolivegreen",0.333f,0.420f,0.184f},
    {"darkorange",1.0f,0.549f,0.0f},{"darkorchid",0.600f,0.196f,0.800f},
    {"darkred",0.545f,0.0f,0.0f},{"darksalmon",0.914f,0.588f,0.478f},
    {"darkseagreen",0.561f,0.737f,0.561f},{"darkslateblue",0.282f,0.239f,0.545f},
    {"darkslategray",0.184f,0.310f,0.310f},{"darkslategrey",0.184f,0.310f,0.310f},
    {"darkturquoise",0.0f,0.808f,0.820f},{"darkviolet",0.580f,0.0f,0.827f},
    {"deeppink",1.0f,0.078f,0.576f},{"deepskyblue",0.0f,0.749f,1.0f},
    {"dimgray",0.412f,0.412f,0.412f},{"dimgrey",0.412f,0.412f,0.412f},
    {"dodgerblue",0.118f,0.565f,1.0f},{"firebrick",0.698f,0.133f,0.133f},
    {"floralwhite",1.0f,0.980f,0.941f},{"forestgreen",0.133f,0.545f,0.133f},
    {"fuchsia",1.0f,0.0f,1.0f},{"gainsboro",0.863f,0.863f,0.863f},
    {"ghostwhite",0.973f,0.973f,1.0f},{"gold",1.0f,0.843f,0.0f},
    {"goldenrod",0.855f,0.647f,0.125f},{"gray",0.502f,0.502f,0.502f},
    {"green",0.0f,0.502f,0.0f},{"greenyellow",0.678f,1.0f,0.184f},
    {"grey",0.502f,0.502f,0.502f},{"honeydew",0.941f,1.0f,0.941f},
    {"hotpink",1.0f,0.412f,0.706f},{"indianred",0.804f,0.361f,0.361f},
    {"indigo",0.294f,0.0f,0.510f},{"ivory",1.0f,1.0f,0.941f},
    {"khaki",0.941f,0.902f,0.549f},{"lavender",0.902f,0.902f,0.980f},
    {"lavenderblush",1.0f,0.941f,0.961f},{"lawngreen",0.486f,0.988f,0.0f},
    {"lemonchiffon",1.0f,0.980f,0.804f},{"lightblue",0.678f,0.847f,0.902f},
    {"lightcoral",0.941f,0.502f,0.502f},{"lightcyan",0.878f,1.0f,1.0f},
    {"lightgoldenrodyellow",0.980f,0.980f,0.824f},
    {"lightgray",0.827f,0.827f,0.827f},{"lightgreen",0.565f,0.933f,0.565f},
    {"lightgrey",0.827f,0.827f,0.827f},{"lightpink",1.0f,0.714f,0.757f},
    {"lightsalmon",1.0f,0.627f,0.478f},{"lightseagreen",0.125f,0.698f,0.667f},
    {"lightskyblue",0.529f,0.808f,0.980f},{"lightslategray",0.467f,0.533f,0.600f},
    {"lightslategrey",0.467f,0.533f,0.600f},{"lightsteelblue",0.690f,0.769f,0.871f},
    {"lightyellow",1.0f,1.0f,0.878f},{"lime",0.0f,1.0f,0.0f},
    {"limegreen",0.196f,0.804f,0.196f},{"linen",0.980f,0.941f,0.902f},
    {"magenta",1.0f,0.0f,1.0f},{"maroon",0.502f,0.0f,0.0f},
    {"mediumaquamarine",0.400f,0.804f,0.667f},{"mediumblue",0.0f,0.0f,0.804f},
    {"mediumorchid",0.729f,0.333f,0.827f},{"mediumpurple",0.576f,0.439f,0.859f},
    {"mediumseagreen",0.235f,0.702f,0.443f},{"mediumslateblue",0.482f,0.408f,0.933f},
    {"mediumspringgreen",0.0f,0.980f,0.604f},{"mediumturquoise",0.282f,0.820f,0.800f},
    {"mediumvioletred",0.780f,0.082f,0.522f},{"midnightblue",0.098f,0.098f,0.439f},
    {"mintcream",0.961f,1.0f,0.980f},{"mistyrose",1.0f,0.894f,0.882f},
    {"moccasin",1.0f,0.894f,0.710f},{"navajowhite",1.0f,0.871f,0.678f},
    {"navy",0.0f,0.0f,0.502f},{"oldlace",0.992f,0.961f,0.902f},
    {"olive",0.502f,0.502f,0.0f},{"olivedrab",0.420f,0.557f,0.137f},
    {"orange",1.0f,0.647f,0.0f},{"orangered",1.0f,0.271f,0.0f},
    {"orchid",0.855f,0.439f,0.839f},{"palegoldenrod",0.933f,0.910f,0.667f},
    {"palegreen",0.596f,0.984f,0.596f},{"paleturquoise",0.686f,0.933f,0.933f},
    {"palevioletred",0.859f,0.439f,0.576f},{"papayawhip",1.0f,0.937f,0.835f},
    {"peachpuff",1.0f,0.855f,0.725f},{"peru",0.804f,0.522f,0.247f},
    {"pink",1.0f,0.753f,0.796f},{"plum",0.867f,0.627f,0.867f},
    {"powderblue",0.690f,0.878f,0.902f},{"purple",0.502f,0.0f,0.502f},
    {"rebeccapurple",0.400f,0.200f,0.600f},{"red",1.0f,0.0f,0.0f},
    {"rosybrown",0.737f,0.561f,0.561f},{"royalblue",0.255f,0.412f,0.882f},
    {"saddlebrown",0.545f,0.271f,0.075f},{"salmon",0.980f,0.502f,0.447f},
    {"sandybrown",0.957f,0.643f,0.376f},{"seagreen",0.180f,0.545f,0.341f},
    {"seashell",1.0f,0.961f,0.933f},{"sienna",0.627f,0.322f,0.176f},
    {"silver",0.753f,0.753f,0.753f},{"skyblue",0.529f,0.808f,0.922f},
    {"slateblue",0.416f,0.353f,0.804f},{"slategray",0.439f,0.502f,0.565f},
    {"slategrey",0.439f,0.502f,0.565f},{"snow",1.0f,0.980f,0.980f},
    {"springgreen",0.0f,1.0f,0.498f},{"steelblue",0.275f,0.510f,0.706f},
    {"tan",0.824f,0.706f,0.549f},{"teal",0.0f,0.502f,0.502f},
    {"thistle",0.847f,0.749f,0.847f},{"tomato",1.0f,0.388f,0.278f},
    {"transparent",0.0f,0.0f,0.0f},{"turquoise",0.251f,0.878f,0.816f},
    {"violet",0.933f,0.510f,0.933f},{"wheat",0.961f,0.871f,0.702f},
    {"white",1.0f,1.0f,1.0f},{"whitesmoke",0.961f,0.961f,0.961f},
    {"yellow",1.0f,1.0f,0.0f},{"yellowgreen",0.604f,0.804f,0.196f},
    {NULL,0,0,0}
};

/* Parse a hex digit (0-15). Returns -1 on failure. */
static int sigil__hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + c - 'a';
    if (c >= 'A' && c <= 'F') return 10 + c - 'A';
    return -1;
}

/* Case-insensitive prefix match of len chars */
static int sigil__ci_prefix(const char *str, int len, const char *prefix) {
    int plen = (int)strlen(prefix);
    if (len < plen) return 0;
    for (int i = 0; i < plen; i++)
        if (tolower((unsigned char)str[i]) != (unsigned char)prefix[i]) return 0;
    return 1;
}

/* Parse comma/space-separated float, advancing *p. Returns 1 on success. */
static int sigil__scan_color_num(const char **p, const char *end, float *val, int *is_pct) {
    while (*p < end && (isspace((unsigned char)**p) || **p == ',' || **p == '/')) (*p)++;
    if (*p >= end) return 0;
    char *ep;
    *val = strtof(*p, &ep);
    if (ep == *p) return 0;
    *p = ep;
    *is_pct = 0;
    if (*p < end && **p == '%') { *is_pct = 1; (*p)++; }
    return 1;
}

/* HSL to RGB conversion */
static void sigil__hsl_to_rgb(float h, float s, float l, float out[3]) {
    h = fmodf(h, 360.0f);
    if (h < 0) h += 360.0f;
    s = s < 0 ? 0 : (s > 1 ? 1 : s);
    l = l < 0 ? 0 : (l > 1 ? 1 : l);
    float c = (1.0f - fabsf(2.0f * l - 1.0f)) * s;
    float x = c * (1.0f - fabsf(fmodf(h / 60.0f, 2.0f) - 1.0f));
    float m = l - c * 0.5f;
    float r1, g1, b1;
    if      (h < 60)  { r1 = c; g1 = x; b1 = 0; }
    else if (h < 120) { r1 = x; g1 = c; b1 = 0; }
    else if (h < 180) { r1 = 0; g1 = c; b1 = x; }
    else if (h < 240) { r1 = 0; g1 = x; b1 = c; }
    else if (h < 300) { r1 = x; g1 = 0; b1 = c; }
    else              { r1 = c; g1 = 0; b1 = x; }
    out[0] = r1 + m; out[1] = g1 + m; out[2] = b1 + m;
}

/* Parse CSS color string into float[4]. Returns:
   1 = valid color, 0 = "none"/"transparent",
   2 = "currentColor", 3 = "inherit" */
#define SIGIL_COLOR_NONE 0
#define SIGIL_COLOR_VALID 1
#define SIGIL_COLOR_CURRENT 2
#define SIGIL_COLOR_INHERIT 3

static int sigil__parse_color(const char *str, int len, float out[4]) {
    out[0] = out[1] = out[2] = 0.0f; out[3] = 1.0f;

    if (len == 0 || !str) return 0;

    /* Skip leading/trailing whitespace */
    while (len > 0 && isspace((unsigned char)*str)) { str++; len--; }
    while (len > 0 && isspace((unsigned char)str[len-1])) { len--; }

    if (len == 4 && sigil__ci_prefix(str, len, "none")) return SIGIL_COLOR_NONE;
    if (len == 11 && sigil__ci_prefix(str, len, "transparent")) {
        out[3] = 0.0f; return SIGIL_COLOR_VALID;
    }
    if (len == 12 && sigil__ci_prefix(str, len, "currentcolor")) return SIGIL_COLOR_CURRENT;
    if (len == 7 && sigil__ci_prefix(str, len, "inherit")) return SIGIL_COLOR_INHERIT;

    /* Hex color: #RGB, #RGBA, #RRGGBB, #RRGGBBAA */
    if (str[0] == '#') {
        if (len == 4) { /* #RGB */
            int r = sigil__hex_digit(str[1]), g = sigil__hex_digit(str[2]), b = sigil__hex_digit(str[3]);
            if (r >= 0 && g >= 0 && b >= 0) {
                out[0] = (float)(r*17)/255.0f; out[1] = (float)(g*17)/255.0f; out[2] = (float)(b*17)/255.0f;
                return 1;
            }
        } else if (len == 5) { /* #RGBA */
            int r = sigil__hex_digit(str[1]), g = sigil__hex_digit(str[2]);
            int b = sigil__hex_digit(str[3]), a = sigil__hex_digit(str[4]);
            if (r >= 0 && g >= 0 && b >= 0 && a >= 0) {
                out[0] = (float)(r*17)/255.0f; out[1] = (float)(g*17)/255.0f;
                out[2] = (float)(b*17)/255.0f; out[3] = (float)(a*17)/255.0f;
                return 1;
            }
        } else if (len == 7) { /* #RRGGBB */
            int r = sigil__hex_digit(str[1])*16 + sigil__hex_digit(str[2]);
            int g = sigil__hex_digit(str[3])*16 + sigil__hex_digit(str[4]);
            int b = sigil__hex_digit(str[5])*16 + sigil__hex_digit(str[6]);
            out[0] = (float)r/255.0f; out[1] = (float)g/255.0f; out[2] = (float)b/255.0f;
            return 1;
        } else if (len == 9) { /* #RRGGBBAA */
            int r = sigil__hex_digit(str[1])*16 + sigil__hex_digit(str[2]);
            int g = sigil__hex_digit(str[3])*16 + sigil__hex_digit(str[4]);
            int b = sigil__hex_digit(str[5])*16 + sigil__hex_digit(str[6]);
            int a = sigil__hex_digit(str[7])*16 + sigil__hex_digit(str[8]);
            out[0] = (float)r/255.0f; out[1] = (float)g/255.0f;
            out[2] = (float)b/255.0f; out[3] = (float)a/255.0f;
            return 1;
        }
    }

    /* rgb() / rgba() — case-insensitive */
    if (sigil__ci_prefix(str, len, "rgb")) {
        const char *p = str + 3;
        const char *end = str + len;
        if (p < end && *p == 'a') p++; /* skip 'a' in rgba */
        if (p < end && *p == '(') p++;
        float rv, gv, bv, av = 1.0f;
        int rp, gp, bp, ap;
        if (sigil__scan_color_num(&p, end, &rv, &rp) &&
            sigil__scan_color_num(&p, end, &gv, &gp) &&
            sigil__scan_color_num(&p, end, &bv, &bp)) {
            /* SVG 1.1: rgb values must be all integers or all percentages.
               Mixed int/percent is invalid. Also, percentage alpha is not supported. */
            if (rp != gp || gp != bp) return 0; /* mixed int/percent = invalid */
            if (rp) { rv = rv * 255.0f / 100.0f; gv = gv * 255.0f / 100.0f; bv = bv * 255.0f / 100.0f; }
            out[0] = (rv < 0 ? 0 : (rv > 255 ? 255 : rv)) / 255.0f;
            out[1] = (gv < 0 ? 0 : (gv > 255 ? 255 : gv)) / 255.0f;
            out[2] = (bv < 0 ? 0 : (bv > 255 ? 255 : bv)) / 255.0f;
            /* Optional alpha */
            if (sigil__scan_color_num(&p, end, &av, &ap)) {
                if (ap) return 0; /* percentage alpha = invalid in SVG context */
                out[3] = av < 0 ? 0 : (av > 1 ? 1 : av);
            }
            return 1;
        }
    }

    /* hsl() / hsla() — case-insensitive */
    if (sigil__ci_prefix(str, len, "hsl")) {
        const char *p = str + 3;
        const char *end = str + len;
        if (p < end && *p == 'a') p++;
        if (p < end && *p == '(') p++;
        float h, s, l, av = 1.0f;
        int hp, sp, lp, ap;
        if (sigil__scan_color_num(&p, end, &h, &hp) &&
            sigil__scan_color_num(&p, end, &s, &sp) &&
            sigil__scan_color_num(&p, end, &l, &lp)) {
            if (sp) s /= 100.0f;
            if (lp) l /= 100.0f;
            sigil__hsl_to_rgb(h, s, l, out);
            if (sigil__scan_color_num(&p, end, &av, &ap)) {
                if (ap) av /= 100.0f;
                out[3] = av < 0 ? 0 : (av > 1 ? 1 : av);
            }
            return 1;
        }
    }

    /* Named colors — case-insensitive */
    {
        char lower[64];
        int clen = len < 63 ? len : 63;
        for (int i = 0; i < clen; i++) lower[i] = (char)tolower((unsigned char)str[i]);
        lower[clen] = '\0';
        for (int i = 0; sigil__named_colors[i].name; i++) {
            if (strcmp(lower, sigil__named_colors[i].name) == 0) {
                out[0] = sigil__named_colors[i].r;
                out[1] = sigil__named_colors[i].g;
                out[2] = sigil__named_colors[i].b;
                if (strcmp(lower, "transparent") == 0) out[3] = 0.0f;
                return 1;
            }
        }
    }

    return 0;
}

/* ------------------------------------------------------------------ */
/*  Transform parsing                                                 */
/* ------------------------------------------------------------------ */

/* 2x3 affine matrix: [a b c d tx ty] representing:
   | a  c  tx |
   | b  d  ty |
   | 0  0  1  |
*/

static void sigil__mat_identity(float m[6]) {
    m[0] = 1; m[1] = 0; m[2] = 0; m[3] = 1; m[4] = 0; m[5] = 0;
}

/* result = A * B (both 2x3 affine) */
static void sigil__mat_multiply(const float A[6], const float B[6], float out[6]) {
    /* A: | a0 a2 a4 |   B: | b0 b2 b4 |
          | a1 a3 a5 |       | b1 b3 b5 |
          | 0  0  1  |       | 0  0  1  | */
    float o0 = A[0]*B[0] + A[2]*B[1];
    float o1 = A[1]*B[0] + A[3]*B[1];
    float o2 = A[0]*B[2] + A[2]*B[3];
    float o3 = A[1]*B[2] + A[3]*B[3];
    float o4 = A[0]*B[4] + A[2]*B[5] + A[4];
    float o5 = A[1]*B[4] + A[3]*B[5] + A[5];
    out[0] = o0; out[1] = o1; out[2] = o2; out[3] = o3; out[4] = o4; out[5] = o5;
}

/* Parse SVG transform attribute */
static void sigil__parse_transform(const char *str, int len, float out[6]) {
    sigil__mat_identity(out);
    if (!str || len <= 0) return;

    /* We need a null-terminated copy for sscanf */
    char *buf = (char *)malloc((size_t)len + 1);
    memcpy(buf, str, (size_t)len);
    buf[len] = '\0';

    const char *p = buf;
    while (*p) {
        while (*p && (isspace((unsigned char)*p) || *p == ',')) p++;
        if (!*p) break;

        float m[6];
        sigil__mat_identity(m);

        if (strncmp(p, "translate", 9) == 0) {
            p += 9;
            while (*p && *p != '(') p++;
            if (*p == '(') p++;
            float tx = 0, ty = 0;
            tx = strtof(p, (char**)&p);
            p = sigil__skip_ws_comma(p);
            if (*p && *p != ')') ty = strtof(p, (char**)&p);
            while (*p && *p != ')') p++;
            if (*p == ')') p++;
            m[4] = tx; m[5] = ty;
        } else if (strncmp(p, "scale", 5) == 0) {
            p += 5;
            while (*p && *p != '(') p++;
            if (*p == '(') p++;
            float sx = 1, sy;
            sx = strtof(p, (char**)&p);
            p = sigil__skip_ws_comma(p);
            if (*p && *p != ')') sy = strtof(p, (char**)&p);
            else sy = sx;
            while (*p && *p != ')') p++;
            if (*p == ')') p++;
            m[0] = sx; m[3] = sy;
        } else if (strncmp(p, "rotate", 6) == 0) {
            p += 6;
            while (*p && *p != '(') p++;
            if (*p == '(') p++;
            float angle = strtof(p, (char**)&p);
            float cx2 = 0, cy2 = 0;
            p = sigil__skip_ws_comma(p);
            if (*p && *p != ')') {
                cx2 = strtof(p, (char**)&p);
                p = sigil__skip_ws_comma(p);
                cy2 = strtof(p, (char**)&p);
            }
            while (*p && *p != ')') p++;
            if (*p == ')') p++;
            float rad = angle * 3.14159265358979f / 180.0f;
            float cosA = cosf(rad), sinA = sinf(rad);
            if (fabsf(cx2) > 1e-6f || fabsf(cy2) > 1e-6f) {
                /* rotate(a, cx, cy) = translate(cx,cy) rotate(a) translate(-cx,-cy) */
                float t1[6] = {1,0,0,1, cx2, cy2};
                float r[6]  = {cosA, sinA, -sinA, cosA, 0, 0};
                float t2[6] = {1,0,0,1, -cx2, -cy2};
                float tmp[6];
                sigil__mat_multiply(t1, r, tmp);
                sigil__mat_multiply(tmp, t2, m);
            } else {
                m[0] = cosA; m[1] = sinA; m[2] = -sinA; m[3] = cosA;
            }
        } else if (strncmp(p, "skewX", 5) == 0) {
            p += 5;
            while (*p && *p != '(') p++;
            if (*p == '(') p++;
            float angle = strtof(p, (char**)&p);
            while (*p && *p != ')') p++;
            if (*p == ')') p++;
            m[2] = tanf(angle * 3.14159265358979f / 180.0f);
        } else if (strncmp(p, "skewY", 5) == 0) {
            p += 5;
            while (*p && *p != '(') p++;
            if (*p == '(') p++;
            float angle = strtof(p, (char**)&p);
            while (*p && *p != ')') p++;
            if (*p == ')') p++;
            m[1] = tanf(angle * 3.14159265358979f / 180.0f);
        } else if (strncmp(p, "matrix", 6) == 0) {
            p += 6;
            while (*p && *p != '(') p++;
            if (*p == '(') p++;
            for (int i = 0; i < 6; i++) {
                p = sigil__skip_ws_comma(p);
                m[i] = strtof(p, (char**)&p);
            }
            while (*p && *p != ')') p++;
            if (*p == ')') p++;
        } else {
            p++;
            continue;
        }

        float tmp[6];
        sigil__mat_multiply(out, m, tmp);
        memcpy(out, tmp, sizeof(float) * 6);
    }

    free(buf);
}

/* ------------------------------------------------------------------ */
/*  Primitive -> curve converters                                     */
/* ------------------------------------------------------------------ */

/* Approximate a quarter of a circle/ellipse using a cubic bezier.
   Uses the standard kappa = 4*(sqrt(2)-1)/3 ≈ 0.5522847498 */
static void sigil__quarter_ellipse(float cx, float cy,
                                    float ax, float ay,
                                    float bx, float by,
                                    SigilCurveArray *arr) {
    float k = 0.5522847498f;
    float c1x = cx + ax + k * bx, c1y = cy + ay + k * by;
    float c2x = cx + bx + k * ax, c2y = cy + by + k * ay;
    float sx = cx + ax, sy = cy + ay;
    float ex = cx + bx, ey = cy + by;
    sigil__cubic_to_quads(sx, sy, c1x, c1y, c2x, c2y, ex, ey, arr);
}

static int sigil__rect_to_curves(float x, float y, float w, float h,
                                  float rx, float ry,
                                  SigilCurve **out, SigilBounds *bounds) {
    SigilCurveArray arr;
    sigil__curve_array_init(&arr);

    if (rx <= 0 && ry <= 0) {
        /* Sharp corners */
        sigil__curve_array_push(&arr, sigil__line_to_quad(x, y, x + w, y));
        sigil__curve_array_push(&arr, sigil__line_to_quad(x + w, y, x + w, y + h));
        sigil__curve_array_push(&arr, sigil__line_to_quad(x + w, y + h, x, y + h));
        sigil__curve_array_push(&arr, sigil__line_to_quad(x, y + h, x, y));
    } else {
        /* Rounded corners using quarter-ellipse arcs.
           Path: top-left corner -> top edge -> top-right corner -> right edge ->
                 bottom-right corner -> bottom edge -> bottom-left corner -> left edge */
        /* Top edge: from (x+rx, y) to (x+w-rx, y) */
        sigil__curve_array_push(&arr, sigil__line_to_quad(x + rx, y, x + w - rx, y));
        /* Top-right corner */
        sigil__quarter_ellipse(x + w - rx, y + ry, 0, -ry, rx, 0, &arr);
        /* Right edge */
        sigil__curve_array_push(&arr, sigil__line_to_quad(x + w, y + ry, x + w, y + h - ry));
        /* Bottom-right corner */
        sigil__quarter_ellipse(x + w - rx, y + h - ry, rx, 0, 0, ry, &arr);
        /* Bottom edge */
        sigil__curve_array_push(&arr, sigil__line_to_quad(x + w - rx, y + h, x + rx, y + h));
        /* Bottom-left corner */
        sigil__quarter_ellipse(x + rx, y + h - ry, 0, ry, -rx, 0, &arr);
        /* Left edge */
        sigil__curve_array_push(&arr, sigil__line_to_quad(x, y + h - ry, x, y + ry));
        /* Top-left corner */
        sigil__quarter_ellipse(x + rx, y + ry, -rx, 0, 0, -ry, &arr);
    }

    bounds->xMin = x; bounds->yMin = y;
    bounds->xMax = x + w; bounds->yMax = y + h;

    *out = arr.data;
    return arr.count;
}

static int sigil__circle_to_curves(float cx, float cy, float r,
                                    SigilCurve **out, SigilBounds *bounds) {
    SigilCurveArray arr;
    sigil__curve_array_init(&arr);

    /* 4 quarter arcs */
    sigil__quarter_ellipse(cx, cy,  0, -r,  r,  0, &arr); /* top to right */
    sigil__quarter_ellipse(cx, cy,  r,  0,  0,  r, &arr); /* right to bottom */
    sigil__quarter_ellipse(cx, cy,  0,  r, -r,  0, &arr); /* bottom to left */
    sigil__quarter_ellipse(cx, cy, -r,  0,  0, -r, &arr); /* left to top */

    bounds->xMin = cx - r; bounds->yMin = cy - r;
    bounds->xMax = cx + r; bounds->yMax = cy + r;

    *out = arr.data;
    return arr.count;
}

static int sigil__ellipse_to_curves(float cx, float cy, float rx, float ry,
                                     SigilCurve **out, SigilBounds *bounds) {
    SigilCurveArray arr;
    sigil__curve_array_init(&arr);

    sigil__quarter_ellipse(cx, cy,  0, -ry,  rx,  0, &arr);
    sigil__quarter_ellipse(cx, cy,  rx,  0,   0, ry, &arr);
    sigil__quarter_ellipse(cx, cy,  0,  ry, -rx,  0, &arr);
    sigil__quarter_ellipse(cx, cy, -rx,  0,   0, -ry, &arr);

    bounds->xMin = cx - rx; bounds->yMin = cy - ry;
    bounds->xMax = cx + rx; bounds->yMax = cy + ry;

    *out = arr.data;
    return arr.count;
}

static int sigil__line_to_curves(float x1, float y1, float x2, float y2,
                                  SigilCurve **out, SigilBounds *bounds) {
    SigilCurveArray arr;
    sigil__curve_array_init(&arr);
    sigil__curve_array_push(&arr, sigil__line_to_quad(x1, y1, x2, y2));

    bounds->xMin = fminf(x1, x2); bounds->yMin = fminf(y1, y2);
    bounds->xMax = fmaxf(x1, x2); bounds->yMax = fmaxf(y1, y2);

    *out = arr.data;
    return arr.count;
}

/* Parse points="x1,y1 x2,y2 ..." into curve array.
   If close_path is true, adds a closing segment. */
static int sigil__polyline_to_curves(const char *points_str, int points_len,
                                      int close_path,
                                      SigilCurve **out, SigilBounds *bounds) {
    SigilCurveArray arr;
    sigil__curve_array_init(&arr);

    /* Need null-terminated copy */
    char *buf = (char *)malloc((size_t)points_len + 1);
    memcpy(buf, points_str, (size_t)points_len);
    buf[points_len] = '\0';

    const char *p = buf;
    float firstX = 0, firstY = 0;
    float prevX = 0, prevY = 0;
    int npoints = 0;

    bounds->xMin = FLT_MAX; bounds->yMin = FLT_MAX;
    bounds->xMax = -FLT_MAX; bounds->yMax = -FLT_MAX;

    while (*p) {
        float x, y;
        const char *np = sigil__parse_float(p, &x);
        if (!np) break;
        p = np;
        np = sigil__parse_float(p, &y);
        if (!np) break;
        p = np;

        if (npoints == 0) {
            firstX = x; firstY = y;
        } else {
            sigil__curve_array_push(&arr, sigil__line_to_quad(prevX, prevY, x, y));
        }
        prevX = x; prevY = y;
        npoints++;

        if (x < bounds->xMin) bounds->xMin = x;
        if (x > bounds->xMax) bounds->xMax = x;
        if (y < bounds->yMin) bounds->yMin = y;
        if (y > bounds->yMax) bounds->yMax = y;
    }

    if (close_path && npoints > 1 &&
        (fabsf(prevX - firstX) > 1e-6f || fabsf(prevY - firstY) > 1e-6f)) {
        sigil__curve_array_push(&arr, sigil__line_to_quad(prevX, prevY, firstX, firstY));
    }

    if (arr.count == 0) {
        bounds->xMin = bounds->yMin = 0;
        bounds->xMax = bounds->yMax = 0;
    }

    free(buf);
    *out = arr.data;
    return arr.count;
}

/* ------------------------------------------------------------------ */
/*  Transform application                                             */
/* ------------------------------------------------------------------ */

static void sigil__transform_point(const float m[6], float *x, float *y) {
    float ox = m[0] * (*x) + m[2] * (*y) + m[4];
    float oy = m[1] * (*x) + m[3] * (*y) + m[5];
    *x = ox; *y = oy;
}

static void sigil__transform_curves(SigilCurve *curves, int count,
                                     const float m[6], SigilBounds *bounds) {
    bounds->xMin = FLT_MAX; bounds->yMin = FLT_MAX;
    bounds->xMax = -FLT_MAX; bounds->yMax = -FLT_MAX;

    for (int i = 0; i < count; i++) {
        sigil__transform_point(m, &curves[i].p0x, &curves[i].p0y);
        sigil__transform_point(m, &curves[i].p1x, &curves[i].p1y);
        sigil__transform_point(m, &curves[i].p2x, &curves[i].p2y);

        float xs[3] = { curves[i].p0x, curves[i].p1x, curves[i].p2x };
        float ys[3] = { curves[i].p0y, curves[i].p1y, curves[i].p2y };
        for (int j = 0; j < 3; j++) {
            if (xs[j] < bounds->xMin) bounds->xMin = xs[j];
            if (xs[j] > bounds->xMax) bounds->xMax = xs[j];
            if (ys[j] < bounds->yMin) bounds->yMin = ys[j];
            if (ys[j] > bounds->yMax) bounds->yMax = ys[j];
        }
    }

    if (count == 0) {
        bounds->xMin = bounds->yMin = 0;
        bounds->xMax = bounds->yMax = 0;
    }
}

/* ------------------------------------------------------------------ */
/*  Band building (uses the same scale+offset formula as slug_wgvk    */
/*  to avoid rounding mismatches between CPU band assignment and      */
/*  GPU pixel lookup)                                                 */
/* ------------------------------------------------------------------ */

static void sigil__build_bands(SigilElement *elem) {
    int nc = (int)elem->curve_count;

    /* Pre-allocate: each band can hold at most all curves */
    for (int b = 0; b < SIGIL_BAND_COUNT; b++) {
        elem->bands.hBands[b].curveIndices = (int *)malloc(nc * sizeof(int));
        elem->bands.hBands[b].count = 0;
        elem->bands.vBands[b].curveIndices = (int *)malloc(nc * sizeof(int));
        elem->bands.vBands[b].count = 0;
    }

    if (nc == 0) return;

    float w = elem->bounds.xMax - elem->bounds.xMin;
    float h = elem->bounds.yMax - elem->bounds.yMin;

    /* Exact same scale+offset as the shader uses for pixel-to-band mapping */
    float bsX = w > 0 ? (float)SIGIL_BAND_COUNT / w : 0;
    float boX = -elem->bounds.xMin * bsX;
    float bsY = h > 0 ? (float)SIGIL_BAND_COUNT / h : 0;
    float boY = -elem->bounds.yMin * bsY;

    for (int i = 0; i < nc; i++) {
        SigilCurve *c = &elem->curves[i];

        float cxMin = fminf(fminf(c->p0x, c->p1x), c->p2x);
        float cxMax = fmaxf(fmaxf(c->p0x, c->p1x), c->p2x);
        float cyMin = fminf(fminf(c->p0y, c->p1y), c->p2y);
        float cyMax = fmaxf(fmaxf(c->p0y, c->p1y), c->p2y);

        if (h > 0) {
            int b0 = (int)fminf(SIGIL_BAND_COUNT - 1, fmaxf(0, floorf(cyMin * bsY + boY)));
            int b1 = (int)fminf(SIGIL_BAND_COUNT - 1, fmaxf(0, floorf(cyMax * bsY + boY)));
            for (int b = b0; b <= b1; b++)
                elem->bands.hBands[b].curveIndices[elem->bands.hBands[b].count++] = i;
        }

        if (w > 0) {
            int b0 = (int)fminf(SIGIL_BAND_COUNT - 1, fmaxf(0, floorf(cxMin * bsX + boX)));
            int b1 = (int)fminf(SIGIL_BAND_COUNT - 1, fmaxf(0, floorf(cxMax * bsX + boX)));
            for (int b = b0; b <= b1; b++)
                elem->bands.vBands[b].curveIndices[elem->bands.vBands[b].count++] = i;
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Stroke to fill conversion                                         */
/* ------------------------------------------------------------------ */

/* Flatten a quadratic Bezier into a polyline by recursive subdivision.
   Appends points (excluding p0, which the caller already added). */
static void sigil__flatten_quad(float p0x, float p0y,
                                float p1x, float p1y,
                                float p2x, float p2y,
                                float **pts, int *npts, int *cap) {
    /* Flatness: distance from p1 to midpoint of p0-p2 */
    float mx = (p0x + p2x) * 0.5f, my = (p0y + p2y) * 0.5f;
    float dx = p1x - mx, dy = p1y - my;

    if (dx * dx + dy * dy <= 0.25f) { /* 0.5 px tolerance */
        /* Flat enough: emit endpoint */
        if (*npts >= *cap) {
            *cap = *cap ? *cap * 2 : 64;
            *pts = (float *)realloc(*pts, (size_t)(*cap) * 2 * sizeof(float));
        }
        (*pts)[(*npts) * 2]     = p2x;
        (*pts)[(*npts) * 2 + 1] = p2y;
        (*npts)++;
        return;
    }

    /* De Casteljau split at t=0.5 */
    float m01x = (p0x + p1x) * 0.5f, m01y = (p0y + p1y) * 0.5f;
    float m12x = (p1x + p2x) * 0.5f, m12y = (p1y + p2y) * 0.5f;
    float midx = (m01x + m12x) * 0.5f, midy = (m01y + m12y) * 0.5f;

    sigil__flatten_quad(p0x, p0y, m01x, m01y, midx, midy, pts, npts, cap);
    sigil__flatten_quad(midx, midy, m12x, m12y, p2x, p2y, pts, npts, cap);
}

/* Push a point into a growable float-pair array */
static void sigil__pts_push(float **pts, int *npts, int *ptcap, float x, float y) {
    if (*npts >= *ptcap) {
        *ptcap = *ptcap ? *ptcap * 2 : 64;
        *pts = (float *)realloc(*pts, (size_t)(*ptcap) * 2 * sizeof(float));
    }
    (*pts)[(*npts) * 2]     = x;
    (*pts)[(*npts) * 2 + 1] = y;
    (*npts)++;
}

/* ---- Join geometry helpers ---- */

static void sigil__join_bevel(SigilCurveArray *arr,
                               float o0x, float o0y,
                               float o1x, float o1y) {
    sigil__curve_array_push(arr, sigil__line_to_quad(o0x, o0y, o1x, o1y));
}

static void sigil__join_miter(SigilCurveArray *arr,
                               float vx, float vy,
                               float n0x, float n0y,
                               float n1x, float n1y,
                               float half, float mlimit,
                               int outer_is_left) {
    float sign = outer_is_left ? 1.0f : -1.0f;
    float o0x = vx + sign * n0x * half, o0y = vy + sign * n0y * half;
    float o1x = vx + sign * n1x * half, o1y = vy + sign * n1y * half;

    float dot_n = n0x * n1x + n0y * n1y;
    float denom = 1.0f + dot_n;

    if (denom < 1e-6f) {
        /* Near-180-degree turn: bevel fallback */
        sigil__join_bevel(arr, o0x, o0y, o1x, o1y);
        return;
    }

    /* Miter limit check: miter_ratio = 1/sin(theta/2),
       theta = angle between path segments at join.
       cos(theta) = -dot(n0,n1), so sin^2(theta/2) = (1 + dot_n)/2 */
    float sin2_half = (1.0f + dot_n) * 0.5f;
    float limit_sq = mlimit * mlimit;
    if (sin2_half > 1e-10f && (1.0f / sin2_half) > limit_sq) {
        sigil__join_bevel(arr, o0x, o0y, o1x, o1y);
        return;
    }

    /* Miter point: V + sign * (n0+n1)/(1+dot(n0,n1)) * half */
    float mx = vx + sign * (n0x + n1x) / denom * half;
    float my = vy + sign * (n0y + n1y) / denom * half;
    sigil__curve_array_push(arr, sigil__line_to_quad(o0x, o0y, mx, my));
    sigil__curve_array_push(arr, sigil__line_to_quad(mx, my, o1x, o1y));
}

static void sigil__join_round(SigilCurveArray *arr,
                               float vx, float vy,
                               float n0x, float n0y,
                               float n1x, float n1y,
                               float half,
                               int outer_is_left) {
    float sign = outer_is_left ? 1.0f : -1.0f;
    float o0x = vx + sign * n0x * half, o0y = vy + sign * n0y * half;

    float dot_n = n0x * n1x + n0y * n1y;
    float cross_n = n0x * n1y - n0y * n1x;
    /* Angle between the two offset directions (on the outer side) */
    float theta = atan2f(sign * cross_n, dot_n);
    /* We want the arc swept on the outer side (reflex direction) */
    if (outer_is_left) {
        if (theta < 0) theta += 2.0f * 3.14159265f;
    } else {
        if (theta > 0) theta -= 2.0f * 3.14159265f;
    }

    float abs_theta = fabsf(theta);
    if (abs_theta < 1e-4f) {
        float o1x = vx + sign * n1x * half, o1y = vy + sign * n1y * half;
        sigil__curve_array_push(arr, sigil__line_to_quad(o0x, o0y, o1x, o1y));
        return;
    }

    /* Split arc into sub-arcs of at most 90 degrees */
    int n_arcs = (int)ceilf(abs_theta / (3.14159265f * 0.5f));
    if (n_arcs < 1) n_arcs = 1;
    float step = theta / (float)n_arcs;

    float angle0 = atan2f(sign * n0y, sign * n0x);
    float prev_px = o0x, prev_py = o0y;

    for (int k = 0; k < n_arcs; k++) {
        float a1 = angle0 + (float)(k + 1) * step;
        float ex = vx + half * cosf(a1);
        float ey = vy + half * sinf(a1);

        /* Control point: push outward along bisector of this sub-arc */
        float a_mid = angle0 + ((float)k + 0.5f) * step;
        float cos_hs = cosf(step * 0.5f);
        if (fabsf(cos_hs) < 1e-6f) cos_hs = (cos_hs < 0) ? -1e-6f : 1e-6f;
        float qr = half / cos_hs;
        float qx = vx + qr * cosf(a_mid);
        float qy = vy + qr * sinf(a_mid);

        SigilCurve c;
        c.p0x = prev_px; c.p0y = prev_py;
        c.p1x = qx;      c.p1y = qy;
        c.p2x = ex;       c.p2y = ey;
        sigil__curve_array_push(arr, c);

        prev_px = ex; prev_py = ey;
    }
}

/* Dispatcher for join geometry on the outer side of a vertex */
static void sigil__emit_join(SigilCurveArray *arr, SigilLineJoin join,
                              float vx, float vy,
                              float n0x, float n0y,
                              float n1x, float n1y,
                              float half, float mlimit,
                              int outer_is_left) {
    switch (join) {
    case SIGIL_JOIN_BEVEL: {
        float sign = outer_is_left ? 1.0f : -1.0f;
        float o0x = vx + sign * n0x * half, o0y = vy + sign * n0y * half;
        float o1x = vx + sign * n1x * half, o1y = vy + sign * n1y * half;
        sigil__join_bevel(arr, o0x, o0y, o1x, o1y);
        break;
    }
    case SIGIL_JOIN_ROUND:
        sigil__join_round(arr, vx, vy, n0x, n0y, n1x, n1y, half, outer_is_left);
        break;
    default: /* SIGIL_JOIN_MITER */
        sigil__join_miter(arr, vx, vy, n0x, n0y, n1x, n1y, half, mlimit, outer_is_left);
        break;
    }
}

/* ---- Cap geometry helpers ---- */

static void sigil__cap_butt(SigilCurveArray *arr,
                             float lx, float ly, float rx, float ry) {
    sigil__curve_array_push(arr, sigil__line_to_quad(lx, ly, rx, ry));
}

static void sigil__cap_square(SigilCurveArray *arr,
                               float lx, float ly, float rx, float ry,
                               float tx, float ty, float half) {
    /* Extend both offset points along the tangent direction */
    float elx = lx + tx * half, ely = ly + ty * half;
    float erx = rx + tx * half, ery = ry + ty * half;
    sigil__curve_array_push(arr, sigil__line_to_quad(lx, ly, elx, ely));
    sigil__curve_array_push(arr, sigil__line_to_quad(elx, ely, erx, ery));
    sigil__curve_array_push(arr, sigil__line_to_quad(erx, ery, rx, ry));
}

static void sigil__cap_round(SigilCurveArray *arr,
                              float vx, float vy,
                              float nx, float ny,
                              float half, int is_start) {
    /* Semicircle: CW arc (sweep = -pi) bulging outward from the path endpoint.
       End cap:   left  -> forward -> right  (outward = tangent direction)
       Start cap: right -> backward -> left  (outward = -tangent direction) */
    float a_start;
    if (is_start) {
        a_start = atan2f(-ny, -nx); /* start from right side */
    } else {
        a_start = atan2f(ny, nx);   /* start from left side */
    }
    float a_end = a_start - 3.14159265f; /* CW sweep */

    /* Two 90-degree quadratic arcs */
    float step = (a_end - a_start) * 0.5f; /* -pi/2 each */
    float cos_hs = cosf(step * 0.5f);
    if (fabsf(cos_hs) < 1e-6f) cos_hs = 1e-6f;
    float qr = half / cos_hs;

    float prev_px = vx + half * cosf(a_start);
    float prev_py = vy + half * sinf(a_start);

    for (int k = 0; k < 2; k++) {
        float a1 = a_start + (float)(k + 1) * step;
        float a_mid = a_start + ((float)k + 0.5f) * step;
        float ex = vx + half * cosf(a1);
        float ey = vy + half * sinf(a1);
        float qx = vx + qr * cosf(a_mid);
        float qy = vy + qr * sinf(a_mid);
        SigilCurve c = { prev_px, prev_py, qx, qy, ex, ey };
        sigil__curve_array_push(arr, c);
        prev_px = ex; prev_py = ey;
    }
}

/* Dispatcher for cap geometry */
static void sigil__emit_cap(SigilCurveArray *arr, SigilLineCap capstyle,
                             float vx, float vy,
                             float nx, float ny,
                             float tx, float ty,
                             float half, int is_start) {
    float lx = vx + nx * half, ly = vy + ny * half;
    float rx = vx - nx * half, ry = vy - ny * half;
    switch (capstyle) {
    case SIGIL_CAP_SQUARE:
        if (is_start)
            sigil__cap_square(arr, rx, ry, lx, ly, -tx, -ty, half);
        else
            sigil__cap_square(arr, lx, ly, rx, ry, tx, ty, half);
        break;
    case SIGIL_CAP_ROUND:
        sigil__cap_round(arr, vx, vy, nx, ny, half, is_start);
        break;
    default: /* SIGIL_CAP_BUTT */
        if (is_start)
            sigil__cap_butt(arr, rx, ry, lx, ly);
        else
            sigil__cap_butt(arr, lx, ly, rx, ry);
        break;
    }
}

/* ---- Inner join: route through vertex for sharp turns, direct for smooth ---- */
static void sigil__inner_join(SigilCurveArray *arr,
                               float vx, float vy,
                               float ipx, float ipy,
                               float inx, float iny) {
    /* For nearly-collinear segments (ip ≈ in), use direct connection to
       avoid thin triangular protrusions that cause Slug rasterizer artifacts.
       For sharp turns, route through the center vertex as before. */
    float dx = inx - ipx, dy = iny - ipy;
    if (dx * dx + dy * dy < 1.0f) {
        /* Points are within ~1 unit — nearly collinear, skip vertex */
        sigil__curve_array_push(arr, sigil__line_to_quad(ipx, ipy, inx, iny));
    } else {
        /* Sharp turn — route through vertex */
        sigil__curve_array_push(arr, sigil__line_to_quad(ipx, ipy, vx, vy));
        sigil__curve_array_push(arr, sigil__line_to_quad(vx, vy, inx, iny));
    }
}

/* ================================================================== */

/* Apply stroke dash array to a set of curves.
   Input: original curves, dash_array (alternating dash/gap lengths), dash_offset.
   Output: new set of curves representing only the dashed segments.
   Each dash becomes a separate sub-path. */
static int sigil__apply_dash(SigilCurve *curves, int count,
                              const float *dash_array, int dash_count,
                              float dash_offset,
                              SigilCurve **out, SigilBounds *bounds) {
    if (count == 0 || dash_count == 0) {
        *out = NULL;
        bounds->xMin = bounds->yMin = bounds->xMax = bounds->yMax = 0;
        return 0;
    }

    /* Flatten curves into polylines per subpath, then walk with dash pattern */
    SigilCurveArray result;
    sigil__curve_array_init(&result);

    /* Detect subpath boundaries */
    int *sp_starts = (int *)malloc((size_t)(count + 1) * sizeof(int));
    int num_sp = 0;
    sp_starts[num_sp++] = 0;
    for (int i = 1; i < count; i++) {
        float dx = curves[i].p0x - curves[i-1].p2x;
        float dy = curves[i].p0y - curves[i-1].p2y;
        if (dx*dx + dy*dy > 1e-6f)
            sp_starts[num_sp++] = i;
    }
    sp_starts[num_sp] = count;

    /* Compute total dash pattern length */
    float dash_total = 0;
    for (int i = 0; i < dash_count; i++) dash_total += dash_array[i];
    if (dash_total <= 0) { free(sp_starts); *out = NULL; return 0; }

    for (int sp = 0; sp < num_sp; sp++) {
        int sp_begin = sp_starts[sp];
        int sp_end = sp_starts[sp + 1];

        /* Flatten subpath */
        float *pts = NULL;
        int npts = 0, ptcap = 0;
        sigil__pts_push(&pts, &npts, &ptcap, curves[sp_begin].p0x, curves[sp_begin].p0y);
        for (int i = sp_begin; i < sp_end; i++)
            sigil__flatten_quad(curves[i].p0x, curves[i].p0y,
                                curves[i].p1x, curves[i].p1y,
                                curves[i].p2x, curves[i].p2y,
                                &pts, &npts, &ptcap);

        if (npts < 2) { free(pts); continue; }

        /* Walk along polyline applying dash pattern */
        float phase = fmodf(dash_offset, dash_total);
        if (phase < 0) phase += dash_total;
        /* Find initial position in dash pattern */
        int dash_idx = 0;
        float dash_remaining = dash_array[0];
        while (phase > 0 && dash_count > 0) {
            if (phase < dash_remaining) {
                dash_remaining -= phase;
                phase = 0;
            } else {
                phase -= dash_remaining;
                dash_idx = (dash_idx + 1) % dash_count;
                dash_remaining = dash_array[dash_idx];
            }
        }
        int is_drawing = (dash_idx % 2 == 0); /* even=dash, odd=gap */

        float prev_x = pts[0], prev_y = pts[1];
        float seg_start_x = prev_x, seg_start_y = prev_y;

        for (int i = 1; i < npts; i++) {
            float nx = pts[i*2], ny = pts[i*2+1];
            float dx = nx - prev_x, dy = ny - prev_y;
            float seg_len = sqrtf(dx*dx + dy*dy);
            if (seg_len < 1e-8f) { prev_x = nx; prev_y = ny; continue; }

            float consumed = 0;
            while (consumed < seg_len - 1e-8f) {
                float remain_in_seg = seg_len - consumed;
                float advance = fminf(remain_in_seg, dash_remaining);
                float t = (consumed + advance) / seg_len;
                float ex = prev_x + t * dx - prev_x * 0 + prev_x;
                /* Interpolate point on segment */
                float px = prev_x + (consumed + advance) / seg_len * dx;
                float py = prev_y + (consumed + advance) / seg_len * dy;

                if (is_drawing) {
                    sigil__curve_array_push(&result, sigil__line_to_quad(
                        seg_start_x, seg_start_y, px, py));
                }

                consumed += advance;
                dash_remaining -= advance;

                if (dash_remaining <= 1e-8f) {
                    dash_idx = (dash_idx + 1) % dash_count;
                    dash_remaining = dash_array[dash_idx];
                    is_drawing = (dash_idx % 2 == 0);
                    seg_start_x = px;
                    seg_start_y = py;
                }
            }

            prev_x = nx;
            prev_y = ny;
        }

        free(pts);
    }

    free(sp_starts);

    /* Compute bounds */
    bounds->xMin = FLT_MAX; bounds->yMin = FLT_MAX;
    bounds->xMax = -FLT_MAX; bounds->yMax = -FLT_MAX;
    for (int i = 0; i < result.count; i++) {
        float xs[3] = { result.data[i].p0x, result.data[i].p1x, result.data[i].p2x };
        float ys[3] = { result.data[i].p0y, result.data[i].p1y, result.data[i].p2y };
        for (int j = 0; j < 3; j++) {
            if (xs[j] < bounds->xMin) bounds->xMin = xs[j];
            if (xs[j] > bounds->xMax) bounds->xMax = xs[j];
            if (ys[j] < bounds->yMin) bounds->yMin = ys[j];
            if (ys[j] > bounds->yMax) bounds->yMax = ys[j];
        }
    }
    if (result.count == 0) {
        bounds->xMin = bounds->yMin = 0;
        bounds->xMax = bounds->yMax = 0;
    }

    *out = result.data;
    return result.count;
}

/* Offset a smooth closed path of quadratic Bezier curves by directly
   offsetting control points along curve normals.  Produces clean ring
   geometry without the polyline flattening artifacts that plague
   sigil__stroke_to_fill on smooth curves like circles and ellipses.
   Returns 0 if the path is not a suitable candidate (caller should
   fall back to sigil__stroke_to_fill). */
static int sigil__stroke_smooth_closed(SigilCurve *curves, int count,
                                        float stroke_width,
                                        SigilCurve **out, SigilBounds *bounds) {
    if (count < 3) return 0; /* need at least 3 curves for a closed shape */
    float half = stroke_width * 0.5f;

    /* Check: single closed subpath with smooth curve connections */
    for (int i = 0; i < count; i++) {
        int next = (i + 1) % count;
        float dx = curves[next].p0x - curves[i].p2x;
        float dy = curves[next].p0y - curves[i].p2y;
        if (dx * dx + dy * dy > 1e-4f) return 0; /* not closed/continuous */

        /* Check curve is a proper quadratic (not a line segment).
           sigil__line_to_quad offsets the midpoint by 0.05 perpendicular,
           so line segments have deviation ≈ 0.0025. Reject those. */
        float mx = (curves[i].p0x + curves[i].p2x) * 0.5f;
        float my = (curves[i].p0y + curves[i].p2y) * 0.5f;
        float ddx = curves[i].p1x - mx;
        float ddy = curves[i].p1y - my;
        if (ddx * ddx + ddy * ddy < 0.01f) return 0; /* line segment */
    }

    SigilCurveArray arr;
    sigil__curve_array_init(&arr);

    /* Forward contour: offset each curve outward (+normal) */
    for (int i = 0; i < count; i++) {
        float t0x = curves[i].p1x - curves[i].p0x;
        float t0y = curves[i].p1y - curves[i].p0y;
        float t1x = curves[i].p2x - curves[i].p1x;
        float t1y = curves[i].p2y - curves[i].p1y;

        float len0 = sqrtf(t0x*t0x + t0y*t0y);
        float len1 = sqrtf(t1x*t1x + t1y*t1y);
        if (len0 < 1e-12f) len0 = 1e-12f;
        if (len1 < 1e-12f) len1 = 1e-12f;

        float n0x = -t0y / len0, n0y = t0x / len0;
        float n1x = -t1y / len1, n1y = t1x / len1;
        /* Average normal at control point */
        float nmx = n0x + n1x, nmy = n0y + n1y;
        float nml = sqrtf(nmx*nmx + nmy*nmy);
        if (nml < 1e-12f) { nmx = n0x; nmy = n0y; }
        else { nmx /= nml; nmy /= nml; }

        SigilCurve c;
        c.p0x = curves[i].p0x + n0x * half;
        c.p0y = curves[i].p0y + n0y * half;
        c.p1x = curves[i].p1x + nmx * half;
        c.p1y = curves[i].p1y + nmy * half;
        c.p2x = curves[i].p2x + n1x * half;
        c.p2y = curves[i].p2y + n1y * half;
        sigil__curve_array_push(&arr, c);
    }

    /* Backward contour: offset each curve inward (-normal), reversed */
    for (int i = count - 1; i >= 0; i--) {
        float t0x = curves[i].p1x - curves[i].p0x;
        float t0y = curves[i].p1y - curves[i].p0y;
        float t1x = curves[i].p2x - curves[i].p1x;
        float t1y = curves[i].p2y - curves[i].p1y;

        float len0 = sqrtf(t0x*t0x + t0y*t0y);
        float len1 = sqrtf(t1x*t1x + t1y*t1y);
        if (len0 < 1e-12f) len0 = 1e-12f;
        if (len1 < 1e-12f) len1 = 1e-12f;

        float n0x = -t0y / len0, n0y = t0x / len0;
        float n1x = -t1y / len1, n1y = t1x / len1;
        float nmx = n0x + n1x, nmy = n0y + n1y;
        float nml = sqrtf(nmx*nmx + nmy*nmy);
        if (nml < 1e-12f) { nmx = n0x; nmy = n0y; }
        else { nmx /= nml; nmy /= nml; }

        /* Reversed curve: swap p0/p2 and negate offset */
        SigilCurve c;
        c.p0x = curves[i].p2x - n1x * half;
        c.p0y = curves[i].p2y - n1y * half;
        c.p1x = curves[i].p1x - nmx * half;
        c.p1y = curves[i].p1y - nmy * half;
        c.p2x = curves[i].p0x - n0x * half;
        c.p2y = curves[i].p0y - n0y * half;
        sigil__curve_array_push(&arr, c);
    }

    /* Compute bounds */
    bounds->xMin = FLT_MAX; bounds->yMin = FLT_MAX;
    bounds->xMax = -FLT_MAX; bounds->yMax = -FLT_MAX;
    for (int i = 0; i < arr.count; i++) {
        float xs[3] = { arr.data[i].p0x, arr.data[i].p1x, arr.data[i].p2x };
        float ys[3] = { arr.data[i].p0y, arr.data[i].p1y, arr.data[i].p2y };
        for (int j = 0; j < 3; j++) {
            if (xs[j] < bounds->xMin) bounds->xMin = xs[j];
            if (xs[j] > bounds->xMax) bounds->xMax = xs[j];
            if (ys[j] < bounds->yMin) bounds->yMin = ys[j];
            if (ys[j] > bounds->yMax) bounds->yMax = ys[j];
        }
    }

    *out = arr.data;
    return arr.count;
}

static int sigil__stroke_to_fill(SigilCurve *curves, int count,
                                  float stroke_width,
                                  SigilLineJoin join, SigilLineCap capstyle,
                                  float miter_limit,
                                  SigilCurve **out, SigilBounds *bounds) {
    if (count == 0 || stroke_width <= 0) {
        *out = NULL;
        bounds->xMin = bounds->yMin = 0;
        bounds->xMax = bounds->yMax = 0;
        return 0;
    }

    float half = stroke_width * 0.5f;
    SigilCurveArray arr;
    sigil__curve_array_init(&arr);

    /* --- Detect subpath boundaries --- */
    int *sp_starts = (int *)malloc((size_t)(count + 1) * sizeof(int));
    int num_sp = 0;
    sp_starts[num_sp++] = 0;
    for (int i = 1; i < count; i++) {
        float dx = curves[i].p0x - curves[i-1].p2x;
        float dy = curves[i].p0y - curves[i-1].p2y;
        if (dx*dx + dy*dy > 1e-6f)
            sp_starts[num_sp++] = i;
    }
    sp_starts[num_sp] = count; /* sentinel */

    /* --- Process each subpath --- */
    for (int sp = 0; sp < num_sp; sp++) {
        int sp_begin = sp_starts[sp];
        int sp_end   = sp_starts[sp + 1];

        /* 1. Flatten all curves of this subpath into one polyline */
        float *pts = NULL;
        int npts = 0, ptcap = 0;

        sigil__pts_push(&pts, &npts, &ptcap,
                        curves[sp_begin].p0x, curves[sp_begin].p0y);
        for (int i = sp_begin; i < sp_end; i++) {
            sigil__flatten_quad(curves[i].p0x, curves[i].p0y,
                                curves[i].p1x, curves[i].p1y,
                                curves[i].p2x, curves[i].p2y,
                                &pts, &npts, &ptcap);
        }

        /* Remove consecutive duplicate points */
        {
            int w = 1;
            for (int r = 1; r < npts; r++) {
                float ddx = pts[r*2] - pts[(w-1)*2];
                float ddy = pts[r*2+1] - pts[(w-1)*2+1];
                if (ddx*ddx + ddy*ddy > 1e-10f) {
                    pts[w*2]   = pts[r*2];
                    pts[w*2+1] = pts[r*2+1];
                    w++;
                }
            }
            npts = w;
        }

        /* 2. Detect closure */
        int is_closed = 0;
        if (npts >= 3) {
            float dcx = pts[0] - pts[(npts-1)*2];
            float dcy = pts[1] - pts[(npts-1)*2+1];
            if (dcx*dcx + dcy*dcy < 1e-4f) {
                is_closed = 1;
                npts--; /* remove duplicate closing point */
            }
        }
        if (npts < 2) { free(pts); continue; }

        int nseg = is_closed ? npts : npts - 1;

        /* 3. Compute per-segment tangent and normal */
        float *seg_tx = (float *)malloc((size_t)nseg * sizeof(float));
        float *seg_ty = (float *)malloc((size_t)nseg * sizeof(float));
        float *seg_nx = (float *)malloc((size_t)nseg * sizeof(float));
        float *seg_ny = (float *)malloc((size_t)nseg * sizeof(float));

        for (int s = 0; s < nseg; s++) {
            int i0 = s;
            int i1 = (s + 1) % npts;
            float tdx = pts[i1*2] - pts[i0*2];
            float tdy = pts[i1*2+1] - pts[i0*2+1];
            float tlen = sqrtf(tdx*tdx + tdy*tdy);
            if (tlen < 1e-12f) tlen = 1e-12f;
            seg_tx[s] = tdx / tlen;
            seg_ty[s] = tdy / tlen;
            seg_nx[s] = -seg_ty[s]; /* 90-degree CCW rotation */
            seg_ny[s] =  seg_tx[s];
        }

        /* 4. Generate stroke outline contour.
           Forward pass (left offset side), then backward pass (right offset side).
           Joins are emitted at each interior vertex. */

        /* --- Forward pass: left side --- */
        for (int s = 0; s < nseg; s++) {
            int i0 = s;
            int i1 = (s + 1) % npts;

            /* Start cap at the beginning of an open path */
            if (s == 0 && !is_closed) {
                sigil__emit_cap(&arr, capstyle,
                                pts[0], pts[1],
                                seg_nx[0], seg_ny[0],
                                seg_tx[0], seg_ty[0],
                                half, 1);
            }

            /* Left offset edge for segment s */
            float l0x = pts[i0*2]   + seg_nx[s] * half;
            float l0y = pts[i0*2+1] + seg_ny[s] * half;
            float l1x = pts[i1*2]   + seg_nx[s] * half;
            float l1y = pts[i1*2+1] + seg_ny[s] * half;
            sigil__curve_array_push(&arr, sigil__line_to_quad(l0x, l0y, l1x, l1y));

            /* Join at vertex i1 to next segment */
            int need_join = 0;
            int s_next = -1;
            if (is_closed) {
                need_join = 1;
                s_next = (s + 1) % nseg;
            } else if (s < nseg - 1) {
                need_join = 1;
                s_next = s + 1;
            }

            if (need_join) {
                float cross = seg_tx[s] * seg_ty[s_next] - seg_ty[s] * seg_tx[s_next];

                if (cross < -1e-6f) {
                    /* Right turn: LEFT side is outer */
                    sigil__emit_join(&arr, join,
                                     pts[i1*2], pts[i1*2+1],
                                     seg_nx[s], seg_ny[s],
                                     seg_nx[s_next], seg_ny[s_next],
                                     half, miter_limit, 1);
                } else if (cross > 1e-6f) {
                    /* Left turn: left side is inner — route through vertex */
                    float ipx = pts[i1*2] + seg_nx[s] * half;
                    float ipy = pts[i1*2+1] + seg_ny[s] * half;
                    float inx = pts[i1*2] + seg_nx[s_next] * half;
                    float iny = pts[i1*2+1] + seg_ny[s_next] * half;
                    sigil__inner_join(&arr, pts[i1*2], pts[i1*2+1],
                                     ipx, ipy, inx, iny);
                }
                /* else collinear: points naturally contiguous */
            }

            /* End cap at the end of an open path */
            if (s == nseg - 1 && !is_closed) {
                int last = is_closed ? (s + 1) % npts : npts - 1;
                sigil__emit_cap(&arr, capstyle,
                                pts[last*2], pts[last*2+1],
                                seg_nx[s], seg_ny[s],
                                seg_tx[s], seg_ty[s],
                                half, 0);
            }
        }

        /* --- Backward pass: right side --- */
        for (int s = nseg - 1; s >= 0; s--) {
            int i0 = s;
            int i1 = (s + 1) % npts;

            /* Right offset edge backward: from i1 to i0 */
            float r1x = pts[i1*2]   - seg_nx[s] * half;
            float r1y = pts[i1*2+1] - seg_ny[s] * half;
            float r0x = pts[i0*2]   - seg_nx[s] * half;
            float r0y = pts[i0*2+1] - seg_ny[s] * half;
            sigil__curve_array_push(&arr, sigil__line_to_quad(r1x, r1y, r0x, r0y));

            /* Join at vertex i0 to the previous segment */
            int need_join = 0;
            int s_prev = -1;
            if (is_closed) {
                need_join = 1;
                s_prev = (s - 1 + nseg) % nseg;
            } else if (s > 0) {
                need_join = 1;
                s_prev = s - 1;
            }

            if (need_join) {
                float cross = seg_tx[s_prev] * seg_ty[s] - seg_ty[s_prev] * seg_tx[s];

                if (cross > 1e-6f) {
                    /* Left turn: RIGHT side is outer */
                    sigil__emit_join(&arr, join,
                                     pts[i0*2], pts[i0*2+1],
                                     seg_nx[s_prev], seg_ny[s_prev],
                                     seg_nx[s], seg_ny[s],
                                     half, miter_limit, 0);
                } else if (cross < -1e-6f) {
                    /* Right turn: right side is inner */
                    float ipx = pts[i0*2] - seg_nx[s_prev] * half;
                    float ipy = pts[i0*2+1] - seg_ny[s_prev] * half;
                    float inx = pts[i0*2] - seg_nx[s] * half;
                    float iny = pts[i0*2+1] - seg_ny[s] * half;
                    sigil__inner_join(&arr, pts[i0*2], pts[i0*2+1],
                                     ipx, ipy, inx, iny);
                }
            }
        }

        free(pts); free(seg_tx); free(seg_ty); free(seg_nx); free(seg_ny);
    }

    free(sp_starts);

    /* Compute bounds */
    bounds->xMin = FLT_MAX; bounds->yMin = FLT_MAX;
    bounds->xMax = -FLT_MAX; bounds->yMax = -FLT_MAX;
    for (int i = 0; i < arr.count; i++) {
        float xs[3] = { arr.data[i].p0x, arr.data[i].p1x, arr.data[i].p2x };
        float ys[3] = { arr.data[i].p0y, arr.data[i].p1y, arr.data[i].p2y };
        for (int j = 0; j < 3; j++) {
            if (xs[j] < bounds->xMin) bounds->xMin = xs[j];
            if (xs[j] > bounds->xMax) bounds->xMax = xs[j];
            if (ys[j] < bounds->yMin) bounds->yMin = ys[j];
            if (ys[j] > bounds->yMax) bounds->yMax = ys[j];
        }
    }

    *out = arr.data;
    return arr.count;
}

/* ------------------------------------------------------------------ */
/*  Element array helper                                              */
/* ------------------------------------------------------------------ */

typedef struct {
    SigilElement *data;
    int count;
    int cap;
} SigilElemArray;

static void sigil__elem_array_init(SigilElemArray *a) {
    a->data = NULL; a->count = 0; a->cap = 0;
}

static SigilElement* sigil__elem_array_push(SigilElemArray *a) {
    if (a->count >= a->cap) {
        a->cap = a->cap ? a->cap * 2 : 16;
        a->data = (SigilElement *)realloc(a->data, (size_t)a->cap * sizeof(SigilElement));
    }
    SigilElement *e = &a->data[a->count++];
    memset(e, 0, sizeof(*e));
    sigil__mat_identity(e->transform);
    e->opacity = 1.0f;
    e->fill_color[3] = 1.0f;
    e->fill_gradient_idx = -1;
    e->stroke_miterlimit = 4.0f;
    return e;
}

/* ------------------------------------------------------------------ */
/*  Helper: check if tag name matches                                 */
/* ------------------------------------------------------------------ */

static int sigil__tag_is(const SigilTag *tag, const char *name) {
    int nlen = (int)strlen(name);
    return tag->name_len == nlen && memcmp(tag->name, name, (size_t)nlen) == 0;
}

/* ------------------------------------------------------------------ */
/*  Helper: parse fill-rule attribute                                 */
/* ------------------------------------------------------------------ */

static SigilFillRule sigil__parse_fill_rule(const char *attrs, int attrs_len) {
    const char *val;
    int vlen = sigil__get_attr(attrs, attrs_len, "fill-rule", &val);
    if (vlen == 7 && memcmp(val, "evenodd", 7) == 0) return SIGIL_FILL_EVENODD;
    return SIGIL_FILL_NONZERO;
}

/* ------------------------------------------------------------------ */
/*  Text support: font loading + glyph curve extraction               */
/* ------------------------------------------------------------------ */

/* Load a font into the scene. Returns 0 on success, -1 on error.
   The font_data must remain valid for the lifetime of the scene.
   This function is only available when SIGIL_IMPLEMENTATION is defined
   (stbtt_fontinfo is not in the public header). */
int sigil_load_font(SigilScene *scene, const char *family_name,
                    const unsigned char *font_data, size_t font_size) {
    if (!scene || !family_name || !font_data || font_size == 0) return -1;

    int idx = scene->font_count;
    scene->fonts = (stbtt_fontinfo *)realloc(scene->fonts,
                    (size_t)(idx + 1) * sizeof(stbtt_fontinfo));
    scene->font_names = (char **)realloc(scene->font_names,
                         (size_t)(idx + 1) * sizeof(char *));
    if (!scene->fonts || !scene->font_names) return -1;

    if (!stbtt_InitFont(&scene->fonts[idx], font_data,
                        stbtt_GetFontOffsetForIndex(font_data, 0))) {
        return -1;
    }
    scene->font_names[idx] = (char *)malloc(strlen(family_name) + 1);
    strcpy(scene->font_names[idx], family_name);
    scene->font_count = idx + 1;
    return 0;
}

/* Extract quadratic bezier curves for a single glyph.
   Same algorithm as slug_wgvk/slug.c extract_curves. */
static int sigil__extract_glyph_curves(const stbtt_fontinfo *font, int glyphIndex,
                                        SigilCurve **out_curves, SigilBounds *out_bounds) {
    stbtt_vertex *verts;
    int nv = stbtt_GetGlyphShape(font, glyphIndex, &verts);
    if (nv <= 0) return 0;

    int ix0, iy0, ix1, iy1;
    if (!stbtt_GetGlyphBox(font, glyphIndex, &ix0, &iy0, &ix1, &iy1)) {
        stbtt_FreeShape(font, verts);
        return 0;
    }
    out_bounds->xMin = (float)ix0;
    out_bounds->yMin = (float)iy0;
    out_bounds->xMax = (float)ix1;
    out_bounds->yMax = (float)iy1;

    SigilCurveArray arr;
    sigil__curve_array_init(&arr);
    float cx = 0, cy = 0;

    for (int i = 0; i < nv; i++) {
        float x = (float)verts[i].x;
        float y = (float)verts[i].y;

        switch (verts[i].type) {
        case STBTT_vmove:
            cx = x; cy = y;
            break;

        case STBTT_vline: {
            float dx = x - cx, dy = y - cy;
            if (fabsf(dx) < 0.1f && fabsf(dy) < 0.1f) { cx = x; cy = y; break; }
            sigil__curve_array_push(&arr, sigil__line_to_quad(cx, cy, x, y));
            cx = x; cy = y;
            break;
        }
        case STBTT_vcurve: {
            SigilCurve c = { cx, cy, (float)verts[i].cx, (float)verts[i].cy, x, y };
            sigil__curve_array_push(&arr, c);
            cx = x; cy = y;
            break;
        }
        case STBTT_vcubic: {
            /* de Casteljau split at t=0.5 -> two quadratics */
            float cx1 = (float)verts[i].cx,  cy1 = (float)verts[i].cy;
            float cx2 = (float)verts[i].cx1, cy2 = (float)verts[i].cy1;
            sigil__cubic_to_quads(cx, cy, cx1, cy1, cx2, cy2, x, y, &arr);
            cx = x; cy = y;
            break;
        }
        }
    }

    stbtt_FreeShape(font, verts);

    *out_curves = arr.data;
    return arr.count;
}

/* Decode one UTF-8 codepoint, advance *p. Returns codepoint or 0 on end. */
static uint32_t sigil__utf8_decode(const char **p, const char *end) {
    if (*p >= end) return 0;
    const unsigned char *s = (const unsigned char *)*p;
    uint32_t cp;
    int len;
    if (s[0] < 0x80)       { cp = s[0]; len = 1; }
    else if (s[0] < 0xC0)  { cp = 0xFFFD; len = 1; } /* invalid continuation */
    else if (s[0] < 0xE0)  { cp = s[0] & 0x1F; len = 2; }
    else if (s[0] < 0xF0)  { cp = s[0] & 0x0F; len = 3; }
    else                    { cp = s[0] & 0x07; len = 4; }
    for (int i = 1; i < len && (s + i < (const unsigned char *)end); i++) {
        if ((s[i] & 0xC0) != 0x80) { *p = (const char *)(s + i); return 0xFFFD; }
        cp = (cp << 6) | (s[i] & 0x3F);
    }
    *p = (const char *)(s + len);
    return cp;
}

/* Convert a text string to curves using stb_truetype.
   scale: pixels per em (from font-size and units_per_em).
   x_pos, y_pos: starting position in SVG coordinates.
   Returns total curve count; *out_curves is malloc'd array. */
static int sigil__text_to_curves(const stbtt_fontinfo *font,
                                  const char *text, int text_len,
                                  float font_size,
                                  float x_pos, float y_pos,
                                  SigilCurve **out_curves, SigilBounds *out_bounds) {
    if (!font || !text || text_len <= 0 || font_size <= 0) {
        *out_curves = NULL;
        out_bounds->xMin = out_bounds->yMin = 0;
        out_bounds->xMax = out_bounds->yMax = 0;
        return 0;
    }

    float scale = stbtt_ScaleForPixelHeight(font, font_size);

    SigilCurveArray all;
    sigil__curve_array_init(&all);

    out_bounds->xMin =  FLT_MAX; out_bounds->yMin =  FLT_MAX;
    out_bounds->xMax = -FLT_MAX; out_bounds->yMax = -FLT_MAX;

    float cursor_x = x_pos;
    /* SVG text y is the baseline; stb_truetype has Y-up in font units,
       so we flip Y and offset by baseline. */
    const char *p = text;
    const char *end = text + text_len;
    uint32_t prev_cp = 0;

    while (p < end) {
        uint32_t cp = sigil__utf8_decode(&p, end);
        if (cp == 0) break;

        int glyph = stbtt_FindGlyphIndex(font, (int)cp);
        if (glyph == 0 && cp != ' ') continue;

        /* Kerning with previous glyph */
        if (prev_cp != 0) {
            int prev_glyph = stbtt_FindGlyphIndex(font, (int)prev_cp);
            int kern = stbtt_GetGlyphKernAdvance(font, prev_glyph, glyph);
            cursor_x += (float)kern * scale;
        }

        /* Extract glyph curves */
        if (glyph != 0) {
            SigilCurve *glyph_curves = NULL;
            SigilBounds glyph_bounds;
            int gc = sigil__extract_glyph_curves(font, glyph, &glyph_curves, &glyph_bounds);

            if (gc > 0 && glyph_curves) {
                /* Transform each curve: scale and position.
                   Font coords: Y-up. SVG: Y-down.
                   Transform: x' = cursor_x + font_x * scale
                              y' = y_pos   - font_y * scale  (flip Y) */
                for (int i = 0; i < gc; i++) {
                    glyph_curves[i].p0x = cursor_x + glyph_curves[i].p0x * scale;
                    glyph_curves[i].p0y = y_pos    - glyph_curves[i].p0y * scale;
                    glyph_curves[i].p1x = cursor_x + glyph_curves[i].p1x * scale;
                    glyph_curves[i].p1y = y_pos    - glyph_curves[i].p1y * scale;
                    glyph_curves[i].p2x = cursor_x + glyph_curves[i].p2x * scale;
                    glyph_curves[i].p2y = y_pos    - glyph_curves[i].p2y * scale;

                    /* Update combined bounds */
                    float xs[3] = { glyph_curves[i].p0x, glyph_curves[i].p1x, glyph_curves[i].p2x };
                    float ys[3] = { glyph_curves[i].p0y, glyph_curves[i].p1y, glyph_curves[i].p2y };
                    for (int j = 0; j < 3; j++) {
                        if (xs[j] < out_bounds->xMin) out_bounds->xMin = xs[j];
                        if (xs[j] > out_bounds->xMax) out_bounds->xMax = xs[j];
                        if (ys[j] < out_bounds->yMin) out_bounds->yMin = ys[j];
                        if (ys[j] > out_bounds->yMax) out_bounds->yMax = ys[j];
                    }

                    sigil__curve_array_push(&all, glyph_curves[i]);
                }
                free(glyph_curves);
            }
        }

        /* Advance cursor */
        int advance, lsb;
        stbtt_GetGlyphHMetrics(font, glyph, &advance, &lsb);
        cursor_x += (float)advance * scale;
        prev_cp = cp;
    }

    if (all.count == 0) {
        out_bounds->xMin = out_bounds->yMin = 0;
        out_bounds->xMax = out_bounds->yMax = 0;
    }

    *out_curves = all.data;
    return all.count;
}

/* Find a font by family name in the scene. Returns pointer or NULL. */
static const stbtt_fontinfo* sigil__find_font(const SigilScene *scene,
                                               const char *family, int family_len) {
    if (!scene || !family || family_len <= 0 || scene->font_count == 0)
        return NULL;
    for (int i = 0; i < scene->font_count; i++) {
        int nlen = (int)strlen(scene->font_names[i]);
        if (nlen == family_len && memcmp(scene->font_names[i], family, (size_t)family_len) == 0)
            return &scene->fonts[i];
    }
    /* If no exact match, return first font as fallback */
    return &scene->fonts[0];
}

/* Extract text content between current position and the </text> closing tag.
   Returns length of text content, fills out_text pointer.
   Advances *pos past the closing </text> tag. */
static int sigil__extract_text_content(const char *src, int src_len, int *pos,
                                        const char **out_text) {
    int start = *pos;
    /* Find </text> */
    const char *close = strstr(src + start, "</text>");
    if (!close) {
        *out_text = NULL;
        return 0;
    }
    int text_len = (int)(close - (src + start));
    *out_text = src + start;
    *pos = (int)(close - src) + 7; /* skip past </text> */
    return text_len;
}

/* ------------------------------------------------------------------ */
/*  Main SVG parser                                                   */
/* ------------------------------------------------------------------ */

SigilScene* sigil_parse_svg(const char* svg_data, size_t len) {
    if (!svg_data || len == 0) {
        return (SigilScene *)calloc(1, sizeof(SigilScene));
    }

    SigilScene *scene = (SigilScene *)calloc(1, sizeof(SigilScene));
    if (!scene) return NULL;

    SigilElemArray elems;
    sigil__elem_array_init(&elems);

    /* Gradient definitions collected during parsing */
    SigilGradientArray grad_defs;
    sigil__grad_array_init(&grad_defs);
    int in_defs = 0;
    int in_gradient = 0; /* index+1 of current gradient being parsed, 0 if none */

    /* Transform stack for <g> nesting */
    float xform_stack[32][6];
    int xform_depth = 0;
    sigil__mat_identity(xform_stack[0]);

    /* Group style stack for inheriting fill/stroke from parent <g> */
    const char *g_style_stack[32] = {NULL};
    int g_style_len_stack[32] = {0};

    /* Use-redirect return stack: when <use> references a container,
       we redirect pos into the container. After the closing tag,
       we restore pos from this stack. Also track target IDs for cycle detection. */
    int use_return_pos[8];
    int use_return_depth[8];
    const char *use_return_id[8];
    int use_return_id_len[8];
    int use_return_n = 0;

    int pos = 0;
    SigilTag tag;

    while (sigil__next_tag(svg_data, (int)len, &pos, &tag)) {
        if (tag.is_close) {
            /* Closing tag */
            if (sigil__tag_is(&tag, "g") || sigil__tag_is(&tag, "svg") || sigil__tag_is(&tag, "symbol")) {
                if (xform_depth > 0) xform_depth--;
                /* Check if this closes a use-redirect container — return to after the <use> */
                if (use_return_n > 0 && xform_depth < use_return_depth[use_return_n - 1]) {
                    use_return_n--;
                    pos = use_return_pos[use_return_n];
                }
            }
            else if (sigil__tag_is(&tag, "defs")) { in_defs = 0; }
            else if (sigil__tag_is(&tag, "linearGradient") ||
                     sigil__tag_is(&tag, "radialGradient")) {
                in_gradient = 0;
                if (xform_depth > 0) xform_depth--; /* pop gradient style frame */
            }
            continue;
        }

        /* <defs> section */
        if (sigil__tag_is(&tag, "defs")) {
            in_defs = 1;
            continue;
        }

        /* <symbol> — like <defs>, content is not rendered directly.
           It's only instantiated when referenced by <use>. */
        if (sigil__tag_is(&tag, "symbol")) {
            if (!tag.self_close) {
                /* Skip until </symbol> */
                int sym_depth = 1;
                SigilTag skip_tag;
                while (sym_depth > 0 && sigil__next_tag(svg_data, (int)len, &pos, &skip_tag)) {
                    if (sigil__tag_is(&skip_tag, "symbol")) {
                        if (skip_tag.is_close) sym_depth--;
                        else if (!skip_tag.self_close) sym_depth++;
                    }
                    /* But still parse gradients inside symbol for potential url() references */
                    if (!skip_tag.is_close && (sigil__tag_is(&skip_tag, "linearGradient") || sigil__tag_is(&skip_tag, "radialGradient"))) {
                        /* Re-inject gradient parsing would be complex; for now, gradients inside symbols
                           must be in <defs> to be found. */
                    }
                }
            }
            continue;
        }

        /* <linearGradient> and <radialGradient> (can appear inside or outside <defs>) */
        if (sigil__tag_is(&tag, "linearGradient")) {
            SigilGradientDef *g = sigil__grad_array_push(&grad_defs);
            g->type = 1;
            g->attrs_set = 0;
            const char *id_val;
            int id_len = sigil__get_attr(tag.attrs, tag.attrs_len, "id", &id_val);
            if (id_len > 0 && id_len < 127) { memcpy(g->id, id_val, (size_t)id_len); g->id[id_len] = '\0'; }
            const char *href_val;
            int href_len = sigil__get_attr(tag.attrs, tag.attrs_len, "xlink:href", &href_val);
            if (href_len == 0) href_len = sigil__get_attr(tag.attrs, tag.attrs_len, "href", &href_val);
            if (href_len > 0 && href_len < 127) { memcpy(g->href, href_val, (size_t)href_len); g->href[href_len] = '\0'; }
            /* gradientUnits */
            const char *gu;
            int gulen = sigil__get_attr(tag.attrs, tag.attrs_len, "gradientUnits", &gu);
            if (gulen > 0) {
                g->attrs_set |= SIGIL_GRAD_HAS_UNITS;
                if (gulen >= 14 && memcmp(gu, "userSpaceOnUse", 14) == 0) g->objectBBox = 0;
            }
            /* spreadMethod */
            const char *sm;
            int smlen = sigil__get_attr(tag.attrs, tag.attrs_len, "spreadMethod", &sm);
            if (smlen > 0) {
                g->attrs_set |= SIGIL_GRAD_HAS_SPREAD;
                if (smlen == 7 && memcmp(sm, "reflect", 7) == 0) g->spread = 1;
                else if (smlen == 6 && memcmp(sm, "repeat", 6) == 0) g->spread = 2;
            }
            /* Endpoint attributes — track which are explicitly set, and whether % was used */
            { const char *v; int vl;
              vl = sigil__get_attr(tag.attrs, tag.attrs_len, "x1", &v);
              if (vl > 0) { g->x1 = strtof(v, NULL); if (v[vl-1]=='%') { g->x1/=100.0f; g->attrs_set |= SIGIL_GRAD_PCT_X1; } g->attrs_set |= SIGIL_GRAD_HAS_X1; }
              vl = sigil__get_attr(tag.attrs, tag.attrs_len, "y1", &v);
              if (vl > 0) { g->y1 = strtof(v, NULL); if (v[vl-1]=='%') { g->y1/=100.0f; g->attrs_set |= SIGIL_GRAD_PCT_Y1; } g->attrs_set |= SIGIL_GRAD_HAS_Y1; }
              vl = sigil__get_attr(tag.attrs, tag.attrs_len, "x2", &v);
              if (vl > 0) { g->x2 = strtof(v, NULL); if (v[vl-1]=='%') { g->x2/=100.0f; g->attrs_set |= SIGIL_GRAD_PCT_X2; } g->attrs_set |= SIGIL_GRAD_HAS_X2; }
              vl = sigil__get_attr(tag.attrs, tag.attrs_len, "y2", &v);
              if (vl > 0) { g->y2 = strtof(v, NULL); if (v[vl-1]=='%') { g->y2/=100.0f; g->attrs_set |= SIGIL_GRAD_PCT_Y2; } g->attrs_set |= SIGIL_GRAD_HAS_Y2; }
            }
            /* gradientTransform */
            const char *gt;
            int gtlen = sigil__get_attr(tag.attrs, tag.attrs_len, "gradientTransform", &gt);
            if (gtlen > 0) { sigil__parse_transform(gt, gtlen, g->transform); g->attrs_set |= SIGIL_GRAD_HAS_XFORM; }
            in_gradient = grad_defs.count;
            /* Push gradient attrs so child <stop> elements can inherit "color" */
            if (!tag.self_close && xform_depth < 31) {
                xform_depth++;
                sigil__mat_identity(xform_stack[xform_depth]);
                g_style_stack[xform_depth] = tag.attrs;
                g_style_len_stack[xform_depth] = tag.attrs_len;
            }
            if (tag.self_close) in_gradient = 0;
            continue;
        }

        if (sigil__tag_is(&tag, "radialGradient")) {
            SigilGradientDef *g = sigil__grad_array_push(&grad_defs);
            g->type = 2;
            g->attrs_set = 0;
            const char *id_val;
            int id_len = sigil__get_attr(tag.attrs, tag.attrs_len, "id", &id_val);
            if (id_len > 0 && id_len < 127) { memcpy(g->id, id_val, (size_t)id_len); g->id[id_len] = '\0'; }
            const char *href_val;
            int href_len = sigil__get_attr(tag.attrs, tag.attrs_len, "xlink:href", &href_val);
            if (href_len == 0) href_len = sigil__get_attr(tag.attrs, tag.attrs_len, "href", &href_val);
            if (href_len > 0 && href_len < 127) { memcpy(g->href, href_val, (size_t)href_len); g->href[href_len] = '\0'; }
            const char *gu;
            int gulen = sigil__get_attr(tag.attrs, tag.attrs_len, "gradientUnits", &gu);
            if (gulen > 0) {
                g->attrs_set |= SIGIL_GRAD_HAS_UNITS;
                if (gulen >= 14 && memcmp(gu, "userSpaceOnUse", 14) == 0) g->objectBBox = 0;
            }
            const char *sm;
            int smlen = sigil__get_attr(tag.attrs, tag.attrs_len, "spreadMethod", &sm);
            if (smlen > 0) {
                g->attrs_set |= SIGIL_GRAD_HAS_SPREAD;
                if (smlen == 7 && memcmp(sm, "reflect", 7) == 0) g->spread = 1;
                else if (smlen == 6 && memcmp(sm, "repeat", 6) == 0) g->spread = 2;
            }
            /* Radial attributes — track which are explicitly set, and whether % was used */
            { const char *v; int vl;
              vl = sigil__get_attr(tag.attrs, tag.attrs_len, "cx", &v);
              if (vl > 0) { g->cx = strtof(v, NULL); if (v[vl-1]=='%') { g->cx/=100.0f; g->attrs_set |= SIGIL_GRAD_PCT_CX; } g->attrs_set |= SIGIL_GRAD_HAS_CX; }
              vl = sigil__get_attr(tag.attrs, tag.attrs_len, "cy", &v);
              if (vl > 0) { g->cy = strtof(v, NULL); if (v[vl-1]=='%') { g->cy/=100.0f; g->attrs_set |= SIGIL_GRAD_PCT_CY; } g->attrs_set |= SIGIL_GRAD_HAS_CY; }
              vl = sigil__get_attr(tag.attrs, tag.attrs_len, "r", &v);
              if (vl > 0) { g->r = strtof(v, NULL); if (v[vl-1]=='%') { g->r/=100.0f; g->attrs_set |= SIGIL_GRAD_PCT_R; } g->attrs_set |= SIGIL_GRAD_HAS_R; }
              vl = sigil__get_attr(tag.attrs, tag.attrs_len, "fx", &v);
              if (vl > 0) { g->fx = strtof(v, NULL); if (v[vl-1]=='%') { g->fx/=100.0f; g->attrs_set |= SIGIL_GRAD_PCT_FX; } g->attrs_set |= SIGIL_GRAD_HAS_FX; }
              else g->fx = -1.0f; /* sentinel */
              vl = sigil__get_attr(tag.attrs, tag.attrs_len, "fy", &v);
              if (vl > 0) { g->fy = strtof(v, NULL); if (v[vl-1]=='%') { g->fy/=100.0f; g->attrs_set |= SIGIL_GRAD_PCT_FY; } g->attrs_set |= SIGIL_GRAD_HAS_FY; }
              else g->fy = -1.0f;
              vl = sigil__get_attr(tag.attrs, tag.attrs_len, "fr", &v);
              if (vl > 0) { g->fr = strtof(v, NULL); if (v[vl-1]=='%') { g->fr/=100.0f; g->attrs_set |= SIGIL_GRAD_PCT_FR; } g->attrs_set |= SIGIL_GRAD_HAS_FR; }
            }
            /* focal defaults to center (done after href resolution) */
            const char *gt;
            int gtlen = sigil__get_attr(tag.attrs, tag.attrs_len, "gradientTransform", &gt);
            if (gtlen > 0) { sigil__parse_transform(gt, gtlen, g->transform); g->attrs_set |= SIGIL_GRAD_HAS_XFORM; }
            in_gradient = grad_defs.count;
            if (!tag.self_close && xform_depth < 31) {
                xform_depth++;
                sigil__mat_identity(xform_stack[xform_depth]);
                g_style_stack[xform_depth] = tag.attrs;
                g_style_len_stack[xform_depth] = tag.attrs_len;
            }
            if (tag.self_close) in_gradient = 0;
            continue;
        }

        /* <stop> inside a gradient */
        if (sigil__tag_is(&tag, "stop") && in_gradient > 0) {
            SigilGradientDef *g = &grad_defs.data[in_gradient - 1];
            g->stops = (SigilGradientStop *)realloc(g->stops,
                (size_t)(g->stop_count + 1) * sizeof(SigilGradientStop));
            SigilGradientStop *s = &g->stops[g->stop_count++];
            memset(s, 0, sizeof(*s));
            s->color[3] = 1.0f;
            /* offset — only number or percentage is valid, ignore units like "5mm" */
            const char *off_val;
            int off_len = sigil__get_attr(tag.attrs, tag.attrs_len, "offset", &off_val);
            if (off_len > 0) {
                char *ep;
                s->offset = strtof(off_val, &ep);
                int rem = off_len - (int)(ep - off_val);
                while (rem > 0 && isspace((unsigned char)*ep)) { ep++; rem--; }
                if (rem > 0 && *ep == '%') { s->offset /= 100.0f; }
                else if (rem > 0 && isalpha((unsigned char)*ep)) {
                    /* Invalid unit (e.g. "5mm") — treat as 0 */
                    s->offset = 0;
                }
            }
            if (s->offset < 0) s->offset = 0;
            if (s->offset > 1) s->offset = 1;

            /* Resolve 'color' property on the stop or gradient for currentColor */
            float stop_current_color[4] = {0, 0, 0, 1};
            {
                const char *cv;
                int cl = sigil__get_attr(tag.attrs, tag.attrs_len, "color", &cv);
                if (cl > 0) {
                    sigil__parse_color(cv, cl, stop_current_color);
                } else {
                    /* Check style for color */
                    const char *ss;
                    int ssl = sigil__get_attr(tag.attrs, tag.attrs_len, "style", &ss);
                    cl = sigil__get_style_prop(ss, ssl, "color", &cv);
                    if (cl > 0) sigil__parse_color(cv, cl, stop_current_color);
                }
                /* Inherit color from parent gradient or groups if not set on stop */
                if (cl == 0) {
                    for (int gi = xform_depth; gi >= 0; gi--) {
                        if (!g_style_stack[gi]) continue;
                        cl = sigil__get_attr(g_style_stack[gi], g_style_len_stack[gi], "color", &cv);
                        if (cl > 0) { sigil__parse_color(cv, cl, stop_current_color); break; }
                    }
                }
            }

            /* stop-color: check style, then attribute */
            const char *sc_style;
            int sc_slen = sigil__get_attr(tag.attrs, tag.attrs_len, "style", &sc_style);
            const char *sc_val;
            int sc_len = sigil__get_prop(tag.attrs, tag.attrs_len, sc_style, sc_slen, "stop-color", &sc_val);
            if (sc_len > 0) {
                int cr = sigil__parse_color(sc_val, sc_len, s->color);
                if (cr == SIGIL_COLOR_CURRENT) {
                    memcpy(s->color, stop_current_color, sizeof(float)*4);
                } else if (cr == SIGIL_COLOR_INHERIT) {
                    /* inherit: look up stop-color from parent gradient element */
                    const char *psc;
                    int pscl = sigil__get_attr(tag.attrs, tag.attrs_len, "stop-color", &psc);
                    if (pscl == 0) {
                        /* Check parent gradient's attrs via the in_gradient mechanism.
                           The gradient is grad_defs.data[in_gradient-1]. We need its tag attrs,
                           but they're not stored. Fall back to the group style stack. */
                        for (int gi = xform_depth; gi >= 0; gi--) {
                            if (!g_style_stack[gi]) continue;
                            pscl = sigil__get_attr(g_style_stack[gi], g_style_len_stack[gi], "stop-color", &psc);
                            if (pscl > 0) { sigil__parse_color(psc, pscl, s->color); break; }
                        }
                    }
                }
            }
            /* stop-opacity: multiply with any alpha from stop-color (e.g. hsla) */
            const char *so_val;
            int so_len = sigil__get_prop(tag.attrs, tag.attrs_len, sc_style, sc_slen, "stop-opacity", &so_val);
            if (so_len > 0) {
                float sop = strtof(so_val, NULL);
                if (so_len > 0 && so_val[so_len-1] == '%') sop /= 100.0f;
                if (sop < 0) sop = 0; if (sop > 1) sop = 1;
                s->color[3] *= sop; /* multiply, don't overwrite */
            }
            continue;
        }

        /* Skip shapes inside <defs> — they are only referenced via <use>/url() */
        if (in_defs) continue;

        /* <svg> tag: extract viewBox, width, height */
        if (sigil__tag_is(&tag, "svg")) {
            /* Track whether this is a nested <svg> vs root */
            int is_root_svg = (scene->width == 0 && scene->height == 0 && !scene->has_viewBox);

            float svg_w = sigil__get_attr_float(tag.attrs, tag.attrs_len, "width", 0);
            float svg_h = sigil__get_attr_float(tag.attrs, tag.attrs_len, "height", 0);

            float svg_vb[4] = {0, 0, 0, 0};
            int has_vb = 0;
            const char *vb;
            int vblen = sigil__get_attr(tag.attrs, tag.attrs_len, "viewBox", &vb);
            if (vblen > 0 && vb) {
                has_vb = 1;
                sscanf(vb, "%f %f %f %f", &svg_vb[0], &svg_vb[1], &svg_vb[2], &svg_vb[3]);
                if (svg_w == 0) svg_w = svg_vb[2];
                if (svg_h == 0) svg_h = svg_vb[3];
            }

            /* Check display:none on the svg element */
            {
                const char *dv;
                int dvlen = sigil__get_attr(tag.attrs, tag.attrs_len, "display", &dv);
                if (dvlen == 4 && memcmp(dv, "none", 4) == 0) {
                    /* Still set scene dimensions before skipping */
                    if (is_root_svg) {
                        scene->width = svg_w > 0 ? svg_w : 300.0f;
                        scene->height = svg_h > 0 ? svg_h : 150.0f;
                        if (has_vb) {
                            scene->has_viewBox = true;
                            memcpy(scene->viewBox, svg_vb, sizeof(svg_vb));
                        }
                    }
                    /* Skip until closing </svg> by tracking depth */
                    int svg_nest = 1;
                    SigilTag skip_tag;
                    while (svg_nest > 0 && sigil__next_tag(svg_data, (int)len, &pos, &skip_tag)) {
                        if (sigil__tag_is(&skip_tag, "svg")) {
                            if (skip_tag.is_close) svg_nest--;
                            else if (!skip_tag.self_close) svg_nest++;
                        }
                    }
                    continue;
                }
            }

            /* Check opacity on the svg element */
            float svg_opacity = 1.0f;
            {
                const char *ov;
                int ol = sigil__get_attr(tag.attrs, tag.attrs_len, "opacity", &ov);
                if (ol > 0) {
                    svg_opacity = strtof(ov, NULL);
                    if (ol > 0 && ov[ol-1] == '%') svg_opacity /= 100.0f;
                    if (svg_opacity < 0) svg_opacity = 0;
                    if (svg_opacity > 1) svg_opacity = 1;
                }
            }

            if (is_root_svg) {
                scene->width = svg_w;
                scene->height = svg_h;
                if (has_vb) {
                    scene->has_viewBox = true;
                    memcpy(scene->viewBox, svg_vb, sizeof(svg_vb));
                }
                /* Set global viewport for CSS units */
                if (svg_w > 0) sigil__vp_width_global = svg_w;
                if (svg_h > 0) sigil__vp_height_global = svg_h;

                /* Handle root SVG opacity by adjusting transform stack */
                if (svg_opacity < 1.0f) {
                    /* Store opacity in a way child elements can inherit it */
                    /* We'll handle this via the g_style_stack mechanism */
                    if (xform_depth < 31) {
                        xform_depth++;
                        sigil__mat_identity(xform_stack[xform_depth]);
                        g_style_stack[xform_depth] = tag.attrs;
                        g_style_len_stack[xform_depth] = tag.attrs_len;
                    }
                }

                /* Handle preserveAspectRatio on root svg */
                if (has_vb && (svg_vb[2] > 0) && (svg_vb[3] > 0)) {
                    const char *par;
                    int par_len = sigil__get_attr(tag.attrs, tag.attrs_len, "preserveAspectRatio", &par);
                    scene->par_align = 5; /* default: xMidYMid */
                    scene->par_meet_or_slice = 0;
                    scene->par_none = 0;
                    if (par_len > 0 && par) {
                        if (par_len >= 4 && memcmp(par, "none", 4) == 0) {
                            scene->par_none = 1;
                        } else if (par_len >= 8) {
                            int xalign = 1, yalign = 1;
                            if (memcmp(par+1, "Min", 3) == 0) xalign = 0;
                            else if (memcmp(par+1, "Mid", 3) == 0) xalign = 1;
                            else if (memcmp(par+1, "Max", 3) == 0) xalign = 2;
                            if (memcmp(par+5, "Min", 3) == 0) yalign = 0;
                            else if (memcmp(par+5, "Mid", 3) == 0) yalign = 1;
                            else if (memcmp(par+5, "Max", 3) == 0) yalign = 2;
                            scene->par_align = yalign * 3 + xalign;
                            const char *ms = par;
                            while (ms < par + par_len && *ms != ' ') ms++;
                            while (ms < par + par_len && *ms == ' ') ms++;
                            int ms_len = par_len - (int)(ms - par);
                            if (ms_len >= 5 && memcmp(ms, "slice", 5) == 0) scene->par_meet_or_slice = 1;
                        }
                    }
                }
            } else {
                /* Nested <svg>: push a new coordinate system */
                float nest_x = sigil__get_attr_float(tag.attrs, tag.attrs_len, "x", 0);
                float nest_y = sigil__get_attr_float(tag.attrs, tag.attrs_len, "y", 0);

                /* Default size: if no width/height, use parent viewport */
                if (svg_w == 0) svg_w = scene->width > 0 ? scene->width : 300.0f;
                if (svg_h == 0) svg_h = scene->height > 0 ? scene->height : 150.0f;

                /* Handle percentage width/height relative to parent viewport */
                {
                    const char *wv;
                    int wl = sigil__get_attr(tag.attrs, tag.attrs_len, "width", &wv);
                    if (wl > 0 && wv[wl-1] == '%') {
                        float pct = strtof(wv, NULL);
                        svg_w = pct * (scene->width > 0 ? scene->width : 300.0f) / 100.0f;
                    }
                    int hl = sigil__get_attr(tag.attrs, tag.attrs_len, "height", &wv);
                    if (hl > 0 && wv[hl-1] == '%') {
                        float pct = strtof(wv, NULL);
                        svg_h = pct * (scene->height > 0 ? scene->height : 150.0f) / 100.0f;
                    }
                }

                /* Build transform: translate(x,y) then viewBox mapping if present */
                float nested_xform[6] = {1, 0, 0, 1, nest_x, nest_y};

                if (has_vb && svg_vb[2] > 0 && svg_vb[3] > 0) {
                    /* Parse preserveAspectRatio */
                    const char *par;
                    int par_len = sigil__get_attr(tag.attrs, tag.attrs_len, "preserveAspectRatio", &par);
                    int align = 6; /* default: xMidYMid */
                    int meet_or_slice = 0; /* 0 = meet (default), 1 = slice */
                    int par_none = 0;

                    if (par_len > 0 && par) {
                        if (par_len >= 4 && memcmp(par, "none", 4) == 0) {
                            par_none = 1;
                        } else {
                            /* Parse alignment: xMinYMin=1, xMidYMin=2, xMaxYMin=3,
                               xMinYMid=4, xMidYMid=5, xMaxYMid=6,
                               xMinYMax=7, xMidYMax=8, xMaxYMax=9 */
                            int xalign = 1, yalign = 1; /* 0=Min, 1=Mid, 2=Max */
                            if (par_len >= 8) {
                                if (memcmp(par+1, "Min", 3) == 0) xalign = 0;
                                else if (memcmp(par+1, "Mid", 3) == 0) xalign = 1;
                                else if (memcmp(par+1, "Max", 3) == 0) xalign = 2;
                                if (memcmp(par+5, "Min", 3) == 0) yalign = 0;
                                else if (memcmp(par+5, "Mid", 3) == 0) yalign = 1;
                                else if (memcmp(par+5, "Max", 3) == 0) yalign = 2;
                            }
                            align = yalign * 3 + xalign;
                            /* Check for "slice" or "meet" */
                            const char *ms = par;
                            while (ms < par + par_len && *ms != ' ') ms++;
                            while (ms < par + par_len && *ms == ' ') ms++;
                            int ms_len = par_len - (int)(ms - par);
                            if (ms_len >= 5 && memcmp(ms, "slice", 5) == 0) meet_or_slice = 1;
                        }
                    }

                    float sx = svg_w / svg_vb[2];
                    float sy = svg_h / svg_vb[3];
                    float tx = -svg_vb[0];
                    float ty = -svg_vb[1];

                    if (!par_none) {
                        float s = meet_or_slice ? fmaxf(sx, sy) : fminf(sx, sy);
                        float dx = svg_w - svg_vb[2] * s;
                        float dy = svg_h - svg_vb[3] * s;
                        int xalign = align % 3, yalign = align / 3;
                        float ox = xalign == 0 ? 0 : (xalign == 1 ? dx * 0.5f : dx);
                        float oy = yalign == 0 ? 0 : (yalign == 1 ? dy * 0.5f : dy);
                        /* transform: translate(nest_x+ox, nest_y+oy) scale(s) translate(-vbx, -vby) */
                        nested_xform[0] = s; nested_xform[1] = 0;
                        nested_xform[2] = 0; nested_xform[3] = s;
                        nested_xform[4] = nest_x + ox + tx * s;
                        nested_xform[5] = nest_y + oy + ty * s;
                    } else {
                        nested_xform[0] = sx; nested_xform[1] = 0;
                        nested_xform[2] = 0;  nested_xform[3] = sy;
                        nested_xform[4] = nest_x + tx * sx;
                        nested_xform[5] = nest_y + ty * sy;
                    }
                }

                if (xform_depth < 31) {
                    xform_depth++;
                    sigil__mat_multiply(xform_stack[xform_depth - 1], nested_xform,
                                        xform_stack[xform_depth]);
                    g_style_stack[xform_depth] = tag.attrs;
                    g_style_len_stack[xform_depth] = tag.attrs_len;
                }
            }
            continue;
        }

        /* <g> tag: push transform */
        if (sigil__tag_is(&tag, "g")) {
            float local[6];
            const char *tr;
            int trlen = sigil__get_attr(tag.attrs, tag.attrs_len, "transform", &tr);
            if (trlen > 0 && tr) {
                sigil__parse_transform(tr, trlen, local);
            } else {
                sigil__mat_identity(local);
            }

            if (xform_depth < 31) {
                xform_depth++;
                sigil__mat_multiply(xform_stack[xform_depth - 1], local,
                                    xform_stack[xform_depth]);
                g_style_stack[xform_depth] = tag.attrs;
                g_style_len_stack[xform_depth] = tag.attrs_len;
            }
            if (tag.self_close && xform_depth > 0) xform_depth--;
            continue;
        }

        /* Shape elements */
        int is_shape = 0;
        int use_shape_frame = 0; /* set when <use> resolves to a shape; pop xform after rendering */
        SigilCurve *curves = NULL;
        int curve_count = 0;
        SigilBounds shape_bounds = {0, 0, 0, 0};

        /* Viewport dimensions for resolving percentage units */
        float vp_w = scene->width > 0 ? scene->width : 200.0f;
        float vp_h = scene->height > 0 ? scene->height : 200.0f;
        float vp_diag = sqrtf(vp_w * vp_w + vp_h * vp_h) / 1.4142136f;

        /* Resolve inherited font-size for em/ex units */
        float font_sz = 16.0f;
        {
            const char *fsv;
            int fsl = sigil__get_attr(tag.attrs, tag.attrs_len, "font-size", &fsv);
            if (fsl > 0) {
                font_sz = sigil__parse_length(fsv, fsl, 16.0f);
            } else {
                for (int gi = xform_depth; gi >= 0; gi--) {
                    if (!g_style_stack[gi]) continue;
                    fsl = sigil__get_attr(g_style_stack[gi], g_style_len_stack[gi], "font-size", &fsv);
                    if (fsl > 0) { font_sz = sigil__parse_length(fsv, fsl, 16.0f); break; }
                }
            }
            sigil__font_size_global = font_sz;
        }

        if (sigil__tag_is(&tag, "path")) {
            is_shape = 1;
            const char *d;
            int dlen = sigil__get_attr(tag.attrs, tag.attrs_len, "d", &d);
            if (dlen > 0 && d) {
                char *dbuf = (char *)malloc((size_t)dlen + 1);
                memcpy(dbuf, d, (size_t)dlen);
                dbuf[dlen] = '\0';
                curve_count = sigil__parse_path(dbuf, &curves, &shape_bounds);
                free(dbuf);
            }
        }
        else if (sigil__tag_is(&tag, "rect")) {
            is_shape = 1;
            float x = sigil__get_attr_vp(tag.attrs, tag.attrs_len, "x", 0, vp_w);
            float y = sigil__get_attr_vp(tag.attrs, tag.attrs_len, "y", 0, vp_h);
            float w = sigil__get_attr_vp(tag.attrs, tag.attrs_len, "width", 0, vp_w);
            float h = sigil__get_attr_vp(tag.attrs, tag.attrs_len, "height", 0, vp_h);
            float rx = sigil__get_attr_vp(tag.attrs, tag.attrs_len, "rx", -1, vp_w);
            float ry = sigil__get_attr_vp(tag.attrs, tag.attrs_len, "ry", -1, vp_h);
            /* SVG spec: rx/ry auto-copy */
            if (rx < 0 && ry >= 0) rx = ry;
            if (ry < 0 && rx >= 0) ry = rx;
            if (rx < 0) rx = 0;
            if (ry < 0) ry = 0;
            /* Clamp rx/ry to half width/height */
            if (rx > w * 0.5f) rx = w * 0.5f;
            if (ry > h * 0.5f) ry = h * 0.5f;
            if (w > 0 && h > 0) {
                curve_count = sigil__rect_to_curves(x, y, w, h, rx, ry,
                                                     &curves, &shape_bounds);
            }
        }
        else if (sigil__tag_is(&tag, "circle")) {
            is_shape = 1;
            float ccx = sigil__get_attr_vp(tag.attrs, tag.attrs_len, "cx", 0, vp_w);
            float ccy = sigil__get_attr_vp(tag.attrs, tag.attrs_len, "cy", 0, vp_h);
            float r = sigil__get_attr_vp(tag.attrs, tag.attrs_len, "r", 0, vp_diag);
            if (r > 0) {
                curve_count = sigil__circle_to_curves(ccx, ccy, r,
                                                       &curves, &shape_bounds);
            }
        }
        else if (sigil__tag_is(&tag, "ellipse")) {
            is_shape = 1;
            float ecx = sigil__get_attr_vp(tag.attrs, tag.attrs_len, "cx", 0, vp_w);
            float ecy = sigil__get_attr_vp(tag.attrs, tag.attrs_len, "cy", 0, vp_h);
            float erx = sigil__get_attr_vp(tag.attrs, tag.attrs_len, "rx", -1, vp_w);
            float ery = sigil__get_attr_vp(tag.attrs, tag.attrs_len, "ry", -1, vp_h);
            /* SVG 2: auto-copy when one radius is missing */
            if (erx < 0 && ery >= 0) erx = ery;
            if (ery < 0 && erx >= 0) ery = erx;
            if (erx < 0) erx = 0;
            if (ery < 0) ery = 0;
            /* Negative radii are an error — don't render */
            if (erx > 0 && ery > 0) {
                curve_count = sigil__ellipse_to_curves(ecx, ecy, erx, ery,
                                                        &curves, &shape_bounds);
            }
        }
        else if (sigil__tag_is(&tag, "line")) {
            is_shape = 1;
            float lx1 = sigil__get_attr_vp(tag.attrs, tag.attrs_len, "x1", 0, vp_w);
            float ly1 = sigil__get_attr_vp(tag.attrs, tag.attrs_len, "y1", 0, vp_h);
            float lx2 = sigil__get_attr_vp(tag.attrs, tag.attrs_len, "x2", 0, vp_w);
            float ly2 = sigil__get_attr_vp(tag.attrs, tag.attrs_len, "y2", 0, vp_h);
            curve_count = sigil__line_to_curves(lx1, ly1, lx2, ly2,
                                                 &curves, &shape_bounds);
        }
        else if (sigil__tag_is(&tag, "polyline")) {
            is_shape = 1;
            const char *pts;
            int ptslen = sigil__get_attr(tag.attrs, tag.attrs_len, "points", &pts);
            if (ptslen > 0 && pts) {
                curve_count = sigil__polyline_to_curves(pts, ptslen, 0,
                                                         &curves, &shape_bounds);
            }
        }
        else if (sigil__tag_is(&tag, "polygon")) {
            is_shape = 1;
            const char *pts;
            int ptslen = sigil__get_attr(tag.attrs, tag.attrs_len, "points", &pts);
            if (ptslen > 0 && pts) {
                curve_count = sigil__polyline_to_curves(pts, ptslen, 1,
                                                         &curves, &shape_bounds);
            }
        }
        else if (sigil__tag_is(&tag, "text") && !tag.self_close) {
            /* <text> element: extract text content and convert to curves */
            float tx = sigil__get_attr_float(tag.attrs, tag.attrs_len, "x", 0);
            float ty = sigil__get_attr_float(tag.attrs, tag.attrs_len, "y", 0);
            float fontSize = sigil__get_attr_float(tag.attrs, tag.attrs_len, "font-size", 16.0f);

            /* Look up font-family */
            const char *family_val;
            int family_len = sigil__get_attr(tag.attrs, tag.attrs_len, "font-family", &family_val);

            const stbtt_fontinfo *font = sigil__find_font(scene, family_val, family_len);

            /* Extract text content between <text> and </text> */
            const char *text_content;
            int text_len = sigil__extract_text_content(svg_data, (int)len, &pos, &text_content);

            if (font && text_content && text_len > 0) {
                is_shape = 1;
                /* Skip leading whitespace in text content */
                while (text_len > 0 && isspace((unsigned char)*text_content)) {
                    text_content++; text_len--;
                }
                while (text_len > 0 && isspace((unsigned char)text_content[text_len - 1])) {
                    text_len--;
                }
                if (text_len > 0) {
                    curve_count = sigil__text_to_curves(font, text_content, text_len,
                                                         fontSize, tx, ty,
                                                         &curves, &shape_bounds);
                }
            }
            /* If no font loaded or no text, skip — don't crash */
        }

        /* <use> element: reference another element by ID */
        if (!is_shape && sigil__tag_is(&tag, "use")) {
            /* Check display:none on the use element itself before processing */
            {
                const char *use_disp;
                const char *use_style;
                int usl = sigil__get_attr(tag.attrs, tag.attrs_len, "style", &use_style);
                int udl = sigil__get_prop(tag.attrs, tag.attrs_len, use_style, usl, "display", &use_disp);
                if (udl == 4 && memcmp(use_disp, "none", 4) == 0) continue;
            }

            const char *href_val;
            int href_len = sigil__get_attr(tag.attrs, tag.attrs_len, "xlink:href", &href_val);
            if (href_len == 0) href_len = sigil__get_attr(tag.attrs, tag.attrs_len, "href", &href_val);
            if (href_len > 1 && href_val[0] == '#') {
                const char *target_id = href_val + 1;
                int target_id_len = href_len - 1;

                /* Get use x/y offset */
                float use_x = sigil__get_attr_float(tag.attrs, tag.attrs_len, "x", 0);
                float use_y = sigil__get_attr_float(tag.attrs, tag.attrs_len, "y", 0);

                /* Scan SVG for element with matching id */
                int scan_pos = 0;
                SigilTag scan_tag;
                while (sigil__next_tag(svg_data, (int)len, &scan_pos, &scan_tag)) {
                    if (scan_tag.is_close) continue;
                    const char *sid;
                    int sid_len = sigil__get_attr(scan_tag.attrs, scan_tag.attrs_len, "id", &sid);
                    if (sid_len == target_id_len && memcmp(sid, target_id, (size_t)sid_len) == 0) {
                        /* Found it. Process as a shape with use's x/y offset */
                        float use_xform[6] = {1, 0, 0, 1, use_x, use_y};
                        float local_xform[6];
                        sigil__mat_multiply(xform_stack[xform_depth], use_xform, local_xform);

                        /* Apply use element's transform too */
                        const char *use_tr;
                        int use_trlen = sigil__get_attr(tag.attrs, tag.attrs_len, "transform", &use_tr);
                        if (use_trlen > 0) {
                            float ut[6];
                            sigil__parse_transform(use_tr, use_trlen, ut);
                            float combined[6];
                            sigil__mat_multiply(local_xform, ut, combined);
                            memcpy(local_xform, combined, sizeof(float)*6);
                        }

                        /* Push a temporary group frame with use's attrs for style inheritance */
                        if (xform_depth < 31) {
                            xform_depth++;
                            memcpy(xform_stack[xform_depth], local_xform, sizeof(float)*6);
                            g_style_stack[xform_depth] = tag.attrs;
                            g_style_len_stack[xform_depth] = tag.attrs_len;
                        }

                        /* Now process the referenced element as a shape */
                        /* We re-inject its tag into the parser loop */
                        SigilCurve *ref_curves = NULL;
                        int ref_count = 0;
                        SigilBounds ref_bounds = {0,0,0,0};
                        int ref_shape = 0;

                        float rvp_w = scene->width > 0 ? scene->width : 200.0f;
                        float rvp_h = scene->height > 0 ? scene->height : 200.0f;

                        if (sigil__tag_is(&scan_tag, "rect")) {
                            ref_shape = 1;
                            float x = sigil__get_attr_vp(scan_tag.attrs, scan_tag.attrs_len, "x", 0, rvp_w);
                            float y = sigil__get_attr_vp(scan_tag.attrs, scan_tag.attrs_len, "y", 0, rvp_h);
                            float w = sigil__get_attr_vp(scan_tag.attrs, scan_tag.attrs_len, "width", 0, rvp_w);
                            float h = sigil__get_attr_vp(scan_tag.attrs, scan_tag.attrs_len, "height", 0, rvp_h);
                            float rx = sigil__get_attr_vp(scan_tag.attrs, scan_tag.attrs_len, "rx", -1, rvp_w);
                            float ry = sigil__get_attr_vp(scan_tag.attrs, scan_tag.attrs_len, "ry", -1, rvp_h);
                            if (rx < 0 && ry >= 0) rx = ry; if (ry < 0 && rx >= 0) ry = rx;
                            if (rx < 0) rx = 0; if (ry < 0) ry = 0;
                            if (rx > w*0.5f) rx = w*0.5f; if (ry > h*0.5f) ry = h*0.5f;
                            if (w > 0 && h > 0) ref_count = sigil__rect_to_curves(x,y,w,h,rx,ry,&ref_curves,&ref_bounds);
                        } else if (sigil__tag_is(&scan_tag, "circle")) {
                            ref_shape = 1;
                            float cx = sigil__get_attr_vp(scan_tag.attrs, scan_tag.attrs_len, "cx", 0, rvp_w);
                            float cy = sigil__get_attr_vp(scan_tag.attrs, scan_tag.attrs_len, "cy", 0, rvp_h);
                            float r = sigil__get_attr_vp(scan_tag.attrs, scan_tag.attrs_len, "r", 0, rvp_w);
                            if (r > 0) ref_count = sigil__circle_to_curves(cx,cy,r,&ref_curves,&ref_bounds);
                        } else if (sigil__tag_is(&scan_tag, "ellipse")) {
                            ref_shape = 1;
                            float cx = sigil__get_attr_vp(scan_tag.attrs, scan_tag.attrs_len, "cx", 0, rvp_w);
                            float cy = sigil__get_attr_vp(scan_tag.attrs, scan_tag.attrs_len, "cy", 0, rvp_h);
                            float rx = sigil__get_attr_vp(scan_tag.attrs, scan_tag.attrs_len, "rx", -1, rvp_w);
                            float ry = sigil__get_attr_vp(scan_tag.attrs, scan_tag.attrs_len, "ry", -1, rvp_h);
                            if (rx < 0 && ry >= 0) rx = ry; if (ry < 0 && rx >= 0) ry = rx;
                            if (rx < 0) rx = 0; if (ry < 0) ry = 0;
                            if (rx > 0 && ry > 0) ref_count = sigil__ellipse_to_curves(cx,cy,rx,ry,&ref_curves,&ref_bounds);
                        } else if (sigil__tag_is(&scan_tag, "line")) {
                            ref_shape = 1;
                            float x1 = sigil__get_attr_vp(scan_tag.attrs, scan_tag.attrs_len, "x1", 0, rvp_w);
                            float y1 = sigil__get_attr_vp(scan_tag.attrs, scan_tag.attrs_len, "y1", 0, rvp_h);
                            float x2 = sigil__get_attr_vp(scan_tag.attrs, scan_tag.attrs_len, "x2", 0, rvp_w);
                            float y2 = sigil__get_attr_vp(scan_tag.attrs, scan_tag.attrs_len, "y2", 0, rvp_h);
                            ref_count = sigil__line_to_curves(x1,y1,x2,y2,&ref_curves,&ref_bounds);
                        } else if (sigil__tag_is(&scan_tag, "path")) {
                            ref_shape = 1;
                            const char *d; int dlen = sigil__get_attr(scan_tag.attrs, scan_tag.attrs_len, "d", &d);
                            if (dlen > 0 && d) {
                                char *dbuf = (char *)malloc((size_t)dlen + 1);
                                memcpy(dbuf, d, (size_t)dlen); dbuf[dlen] = '\0';
                                ref_count = sigil__parse_path(dbuf, &ref_curves, &ref_bounds);
                                free(dbuf);
                            }
                        } else if (sigil__tag_is(&scan_tag, "polygon")) {
                            ref_shape = 1;
                            const char *pts; int ptslen = sigil__get_attr(scan_tag.attrs, scan_tag.attrs_len, "points", &pts);
                            if (ptslen > 0) ref_count = sigil__polyline_to_curves(pts, ptslen, 1, &ref_curves, &ref_bounds);
                        } else if (sigil__tag_is(&scan_tag, "polyline")) {
                            ref_shape = 1;
                            const char *pts; int ptslen = sigil__get_attr(scan_tag.attrs, scan_tag.attrs_len, "points", &pts);
                            if (ptslen > 0) ref_count = sigil__polyline_to_curves(pts, ptslen, 0, &ref_curves, &ref_bounds);
                        } else if (sigil__tag_is(&scan_tag, "g") ||
                                   sigil__tag_is(&scan_tag, "symbol") ||
                                   sigil__tag_is(&scan_tag, "svg")) {
                            /* <use> referencing a container: redirect main parser
                               into the container's children, with a return address
                               so we jump back after the closing tag. */

                            /* Cycle detection: check if we're already inside this container */
                            int use_cycle = 0;
                            for (int ri = 0; ri < use_return_n; ri++) {
                                if (use_return_id_len[ri] == target_id_len &&
                                    memcmp(use_return_id[ri], target_id, (size_t)target_id_len) == 0) {
                                    use_cycle = 1; break;
                                }
                            }
                            if (use_cycle || use_return_n >= 8) break; /* bail on cycle or too deep */

                            int is_sym = sigil__tag_is(&scan_tag, "symbol") || sigil__tag_is(&scan_tag, "svg");

                            /* Apply symbol/svg viewBox mapping */
                            if (is_sym) {
                                const char *svb;
                                int svbl = sigil__get_attr(scan_tag.attrs, scan_tag.attrs_len, "viewBox", &svb);
                                if (svbl > 0) {
                                    float svb4[4] = {0,0,0,0};
                                    sscanf(svb, "%f %f %f %f", &svb4[0], &svb4[1], &svb4[2], &svb4[3]);
                                    /* Use width/height default to parent viewport, not viewBox size */
                                    float def_w = scene->width > 0 ? scene->width : 200.0f;
                                    float def_h = scene->height > 0 ? scene->height : 200.0f;
                                    float uw = sigil__get_attr_float(tag.attrs, tag.attrs_len, "width", def_w);
                                    float uh = sigil__get_attr_float(tag.attrs, tag.attrs_len, "height", def_h);
                                    if (uw <= 0) uw = def_w; if (uh <= 0) uh = def_h;
                                    if (svb4[2] > 0 && svb4[3] > 0) {
                                        float sx = uw / svb4[2], sy = uh / svb4[3];
                                        float s = fminf(sx, sy); /* default: xMidYMid meet */
                                        float dx = uw - svb4[2] * s, dy = uh - svb4[3] * s;
                                        float vb_xform[6] = {s, 0, 0, s,
                                            -svb4[0]*s + dx*0.5f,
                                            -svb4[1]*s + dy*0.5f};
                                        float combined[6];
                                        sigil__mat_multiply(xform_stack[xform_depth], vb_xform, combined);
                                        memcpy(xform_stack[xform_depth], combined, sizeof(float)*6);
                                    }
                                }
                            }

                            /* Apply the container's own transform — but NOT for <symbol>
                               (SVG 1.1: symbol elements don't support transform attribute) */
                            if (!sigil__tag_is(&scan_tag, "symbol")) {
                                const char *gtr;
                                int gtrl = sigil__get_attr(scan_tag.attrs, scan_tag.attrs_len, "transform", &gtr);
                                if (gtrl > 0) {
                                    float gt[6]; sigil__parse_transform(gtr, gtrl, gt);
                                    float combined[6];
                                    sigil__mat_multiply(xform_stack[xform_depth], gt, combined);
                                    memcpy(xform_stack[xform_depth], combined, sizeof(float)*6);
                                }
                            }

                            g_style_stack[xform_depth] = scan_tag.attrs;
                            g_style_len_stack[xform_depth] = scan_tag.attrs_len;

                            /* Save return address and target ID for cycle detection */
                            if (use_return_n < 8) {
                                use_return_pos[use_return_n] = pos;
                                use_return_depth[use_return_n] = xform_depth;
                                use_return_id[use_return_n] = target_id;
                                use_return_id_len[use_return_n] = target_id_len;
                                use_return_n++;
                            }
                            pos = scan_pos;
                            /* ref_shape stays 0; is_shape stays 0;
                               main loop continues from children.
                               Closing tag handler returns us after the <use>. */
                            break;
                        } else if (sigil__tag_is(&scan_tag, "use")) {
                            /* Indirect <use>: resolve chain with cycle detection */
                            const char *visited[10];
                            int visited_len[10];
                            int visited_n = 0;
                            visited[visited_n] = target_id;
                            visited_len[visited_n] = target_id_len;
                            visited_n++;

                            SigilTag chain_tag = scan_tag;
                            while (visited_n < 10) {
                                const char *chr;
                                int chrl = sigil__get_attr(chain_tag.attrs, chain_tag.attrs_len, "xlink:href", &chr);
                                if (chrl == 0) chrl = sigil__get_attr(chain_tag.attrs, chain_tag.attrs_len, "href", &chr);
                                if (chrl <= 1 || chr[0] != '#') break;
                                const char *next_id = chr + 1;
                                int next_id_len = chrl - 1;

                                /* Cycle detection */
                                int is_cycle = 0;
                                for (int v = 0; v < visited_n; v++) {
                                    if (visited_len[v] == next_id_len &&
                                        memcmp(visited[v], next_id, (size_t)next_id_len) == 0) {
                                        is_cycle = 1; break;
                                    }
                                }
                                if (is_cycle) break;
                                visited[visited_n] = next_id;
                                visited_len[visited_n] = next_id_len;
                                visited_n++;

                                /* Apply intermediate use's x/y */
                                float ix = sigil__get_attr_float(chain_tag.attrs, chain_tag.attrs_len, "x", 0);
                                float iy = sigil__get_attr_float(chain_tag.attrs, chain_tag.attrs_len, "y", 0);
                                if (fabsf(ix) > 1e-6f || fabsf(iy) > 1e-6f) {
                                    float it[6] = {1,0,0,1, ix, iy};
                                    float combined[6];
                                    sigil__mat_multiply(xform_stack[xform_depth], it, combined);
                                    memcpy(xform_stack[xform_depth], combined, sizeof(float)*6);
                                }

                                /* Scan for the next target */
                                int scan2 = 0;
                                SigilTag scan2_tag;
                                int found_next = 0;
                                while (sigil__next_tag(svg_data, (int)len, &scan2, &scan2_tag)) {
                                    if (scan2_tag.is_close) continue;
                                    const char *sid2;
                                    int sid2l = sigil__get_attr(scan2_tag.attrs, scan2_tag.attrs_len, "id", &sid2);
                                    if (sid2l == next_id_len && memcmp(sid2, next_id, (size_t)sid2l) == 0) {
                                        if (sigil__tag_is(&scan2_tag, "use")) {
                                            chain_tag = scan2_tag;
                                            found_next = 1;
                                        } else {
                                            /* Final target: process as shape */
                                            if (sigil__tag_is(&scan2_tag, "rect")) {
                                                ref_shape = 1;
                                                float x = sigil__get_attr_vp(scan2_tag.attrs, scan2_tag.attrs_len, "x", 0, rvp_w);
                                                float y = sigil__get_attr_vp(scan2_tag.attrs, scan2_tag.attrs_len, "y", 0, rvp_h);
                                                float w = sigil__get_attr_vp(scan2_tag.attrs, scan2_tag.attrs_len, "width", 0, rvp_w);
                                                float h = sigil__get_attr_vp(scan2_tag.attrs, scan2_tag.attrs_len, "height", 0, rvp_h);
                                                float rrx = sigil__get_attr_float(scan2_tag.attrs, scan2_tag.attrs_len, "rx", -1);
                                                float rry = sigil__get_attr_float(scan2_tag.attrs, scan2_tag.attrs_len, "ry", -1);
                                                if (rrx < 0 && rry >= 0) rrx = rry; if (rry < 0 && rrx >= 0) rry = rrx;
                                                if (rrx < 0) rrx = 0; if (rry < 0) rry = 0;
                                                if (w > 0 && h > 0) ref_count = sigil__rect_to_curves(x,y,w,h,rrx,rry,&ref_curves,&ref_bounds);
                                            } else if (sigil__tag_is(&scan2_tag, "circle")) {
                                                ref_shape = 1;
                                                float cx = sigil__get_attr_vp(scan2_tag.attrs, scan2_tag.attrs_len, "cx", 0, rvp_w);
                                                float cy = sigil__get_attr_vp(scan2_tag.attrs, scan2_tag.attrs_len, "cy", 0, rvp_h);
                                                float r = sigil__get_attr_vp(scan2_tag.attrs, scan2_tag.attrs_len, "r", 0, rvp_w);
                                                if (r > 0) ref_count = sigil__circle_to_curves(cx,cy,r,&ref_curves,&ref_bounds);
                                            } else if (sigil__tag_is(&scan2_tag, "ellipse")) {
                                                ref_shape = 1;
                                                float cx = sigil__get_attr_vp(scan2_tag.attrs, scan2_tag.attrs_len, "cx", 0, rvp_w);
                                                float cy = sigil__get_attr_vp(scan2_tag.attrs, scan2_tag.attrs_len, "cy", 0, rvp_h);
                                                float erx = sigil__get_attr_vp(scan2_tag.attrs, scan2_tag.attrs_len, "rx", -1, rvp_w);
                                                float ery = sigil__get_attr_vp(scan2_tag.attrs, scan2_tag.attrs_len, "ry", -1, rvp_h);
                                                if (erx < 0 && ery >= 0) erx = ery; if (ery < 0 && erx >= 0) ery = erx;
                                                if (erx < 0) erx = 0; if (ery < 0) ery = 0;
                                                if (erx > 0 && ery > 0) ref_count = sigil__ellipse_to_curves(cx,cy,erx,ery,&ref_curves,&ref_bounds);
                                            } else if (sigil__tag_is(&scan2_tag, "line")) {
                                                ref_shape = 1;
                                                float x1 = sigil__get_attr_vp(scan2_tag.attrs, scan2_tag.attrs_len, "x1", 0, rvp_w);
                                                float y1 = sigil__get_attr_vp(scan2_tag.attrs, scan2_tag.attrs_len, "y1", 0, rvp_h);
                                                float x2 = sigil__get_attr_vp(scan2_tag.attrs, scan2_tag.attrs_len, "x2", 0, rvp_w);
                                                float y2 = sigil__get_attr_vp(scan2_tag.attrs, scan2_tag.attrs_len, "y2", 0, rvp_h);
                                                ref_count = sigil__line_to_curves(x1,y1,x2,y2,&ref_curves,&ref_bounds);
                                            } else if (sigil__tag_is(&scan2_tag, "path")) {
                                                ref_shape = 1;
                                                const char *d; int dlen = sigil__get_attr(scan2_tag.attrs, scan2_tag.attrs_len, "d", &d);
                                                if (dlen > 0 && d) {
                                                    char *dbuf = (char *)malloc((size_t)dlen + 1);
                                                    memcpy(dbuf, d, (size_t)dlen); dbuf[dlen] = '\0';
                                                    ref_count = sigil__parse_path(dbuf, &ref_curves, &ref_bounds);
                                                    free(dbuf);
                                                }
                                            } else if (sigil__tag_is(&scan2_tag, "polygon")) {
                                                ref_shape = 1;
                                                const char *pts; int ptslen = sigil__get_attr(scan2_tag.attrs, scan2_tag.attrs_len, "points", &pts);
                                                if (ptslen > 0) ref_count = sigil__polyline_to_curves(pts, ptslen, 1, &ref_curves, &ref_bounds);
                                            } else if (sigil__tag_is(&scan2_tag, "polyline")) {
                                                ref_shape = 1;
                                                const char *pts; int ptslen = sigil__get_attr(scan2_tag.attrs, scan2_tag.attrs_len, "points", &pts);
                                                if (ptslen > 0) ref_count = sigil__polyline_to_curves(pts, ptslen, 0, &ref_curves, &ref_bounds);
                                            }
                                            if (ref_shape) scan_tag = scan2_tag;
                                        }
                                        break;
                                    }
                                }
                                if (!found_next || ref_shape) break;
                            }
                        }

                        if (ref_shape && ref_count > 0) {
                            is_shape = 1;
                            use_shape_frame = 1;
                            curves = ref_curves;
                            curve_count = ref_count;
                            shape_bounds = ref_bounds;
                            tag = scan_tag;
                        } else if (use_return_n == 0 ||
                                   use_return_depth[use_return_n-1] != xform_depth) {
                            /* Only pop use frame if we didn't set up a container redirect */
                            free(ref_curves);
                            if (xform_depth > 0) xform_depth--;
                        }
                        break;
                    }
                }
            }
        }

        if (!is_shape) continue;

        /* Extract inline style attribute for CSS property lookups */
        const char *style_str;
        int style_len = sigil__get_attr(tag.attrs, tag.attrs_len, "style", &style_str);

        /* Check display/visibility (including group inheritance) */
        {
            const char *dv;
            int dvlen = sigil__get_prop(tag.attrs, tag.attrs_len, style_str, style_len, "display", &dv);
            if (dvlen == 4 && memcmp(dv, "none", 4) == 0) { free(curves); continue; }

            /* Check visibility on element and parent groups */
            dvlen = sigil__get_prop(tag.attrs, tag.attrs_len, style_str, style_len, "visibility", &dv);
            if (dvlen == 6 && memcmp(dv, "hidden", 6) == 0) { free(curves); continue; }
            if (dvlen == 8 && memcmp(dv, "collapse", 8) == 0) { free(curves); continue; }
            /* Inherit visibility from parent groups */
            int inherited_hidden = 0;
            if (dvlen == 0) {
                for (int gi = xform_depth; gi >= 0; gi--) {
                    if (!g_style_stack[gi]) continue;
                    dvlen = sigil__get_attr(g_style_stack[gi], g_style_len_stack[gi], "visibility", &dv);
                    if (dvlen > 0) {
                        if ((dvlen == 6 && memcmp(dv, "hidden", 6) == 0) ||
                            (dvlen == 8 && memcmp(dv, "collapse", 8) == 0))
                        { inherited_hidden = 1; }
                        break;
                    }
                }
            }
            if (inherited_hidden) { free(curves); continue; }
        }

        /* Resolve currentColor: look at element's "color" prop, then parent groups */
        float current_color[4] = {0, 0, 0, 1};
        {
            const char *cc_val;
            int cc_len = sigil__get_prop(tag.attrs, tag.attrs_len, style_str, style_len, "color", &cc_val);
            int cr = 0;
            if (cc_len > 0) {
                cr = sigil__parse_color(cc_val, cc_len, current_color);
            }
            /* If not set or inherit, walk parent groups */
            if (cc_len == 0 || cr == SIGIL_COLOR_INHERIT) {
                for (int gi = xform_depth; gi >= 0; gi--) {
                    if (!g_style_stack[gi]) continue;
                    cc_len = sigil__get_attr(g_style_stack[gi], g_style_len_stack[gi], "color", &cc_val);
                    if (cc_len > 0) {
                        int gcr = sigil__parse_color(cc_val, cc_len, current_color);
                        if (gcr != SIGIL_COLOR_INHERIT) break;
                        /* else continue walking up */
                    }
                }
            }
        }

        /* Extract style attributes (style overrides presentation attributes) */
        float fill_color[4] = {0, 0, 0, 1}; /* default fill = black */
        float stroke_color[4] = {0, 0, 0, 0};
        float stroke_width = 0;
        float opacity = 1.0f;
        float fill_opacity = 1.0f;
        float stroke_opacity = 1.0f;
        SigilFillRule fill_rule = SIGIL_FILL_NONZERO;
        int has_fill = 1;
        int has_stroke = 0;
        int fill_gradient_idx = -1;
        int stroke_gradient_idx = -1;

        /* <line> elements have no interior area; default fill to none */
        if (sigil__tag_is(&tag, "line")) has_fill = 0;

        const char *fill_val;
        int fill_vlen = sigil__get_prop(tag.attrs, tag.attrs_len, style_str, style_len, "fill", &fill_val);

        /* Try resolving url(#id) with fallback color */
        if (fill_vlen > 4 && memcmp(fill_val, "url(", 4) == 0) {
            has_fill = 0;
            const char *hash = (const char*)memchr(fill_val, '#', (size_t)fill_vlen);
            if (hash && grad_defs.count > 0) {
                hash++;
                const char *end_paren = (const char*)memchr(hash, ')', (size_t)(fill_vlen - (int)(hash - fill_val)));
                int id_len = end_paren ? (int)(end_paren - hash) : (int)(fill_vlen - (int)(hash - fill_val));
                /* Trim whitespace and optional quotes from id */
                while (id_len > 0 && isspace((unsigned char)hash[id_len-1])) id_len--;
                while (id_len > 0 && (hash[id_len-1] == '\'' || hash[id_len-1] == '"')) id_len--;
                for (int gi = 0; gi < grad_defs.count; gi++) {
                    if ((int)strlen(grad_defs.data[gi].id) == id_len &&
                        memcmp(grad_defs.data[gi].id, hash, (size_t)id_len) == 0) {
                        /* Accept if has stops or has href (stops inherited later) */
                        if (grad_defs.data[gi].stop_count > 0 ||
                            grad_defs.data[gi].href[0] != '\0') {
                            fill_gradient_idx = gi;
                            has_fill = 1;
                        }
                        break;
                    }
                }
            }
            /* Fallback color after closing paren: "url(#id) green" */
            if (!has_fill) {
                const char *cp = (const char*)memchr(fill_val, ')', (size_t)fill_vlen);
                if (cp) {
                    cp++;
                    int rem = fill_vlen - (int)(cp - fill_val);
                    while (rem > 0 && isspace((unsigned char)*cp)) { cp++; rem--; }
                    if (rem > 0) {
                        int cr = sigil__parse_color(cp, rem, fill_color);
                        if (cr == SIGIL_COLOR_CURRENT) { memcpy(fill_color, current_color, 16); cr = 1; }
                        has_fill = (cr == SIGIL_COLOR_VALID);
                    }
                }
            }
        } else if (fill_vlen > 0) {
            int cr = sigil__parse_color(fill_val, fill_vlen, fill_color);
            if (cr == SIGIL_COLOR_CURRENT) {
                memcpy(fill_color, current_color, sizeof(float)*4); has_fill = 1;
            } else if (cr == SIGIL_COLOR_INHERIT) {
                has_fill = 0;
                for (int gi = xform_depth; gi >= 0; gi--) {
                    if (!g_style_stack[gi]) continue;
                    const char *gf;
                    int gflen = sigil__get_attr(g_style_stack[gi], g_style_len_stack[gi], "fill", &gf);
                    if (gflen > 0) {
                        int gcr = sigil__parse_color(gf, gflen, fill_color);
                        if (gcr == SIGIL_COLOR_CURRENT) { memcpy(fill_color, current_color, 16); has_fill = 1; break; }
                        else if (gcr == SIGIL_COLOR_INHERIT) continue; /* keep walking up */
                        else if (gcr == SIGIL_COLOR_VALID) { has_fill = 1; break; }
                        else { has_fill = 0; break; } /* none */
                    }
                }
                /* If no parent had fill, use initial value (black) */
                if (!has_fill) {
                    int found_any = 0;
                    for (int gi = xform_depth; gi >= 0; gi--) {
                        if (g_style_stack[gi]) {
                            const char *gf; int gflen = sigil__get_attr(g_style_stack[gi], g_style_len_stack[gi], "fill", &gf);
                            if (gflen > 0) { found_any = 1; break; }
                        }
                    }
                    if (!found_any) {
                        fill_color[0] = 0; fill_color[1] = 0; fill_color[2] = 0; fill_color[3] = 1;
                        has_fill = 1; /* inherit with no parent = initial value (black) */
                    }
                }
            } else if (cr == SIGIL_COLOR_NONE) {
                /* Check if explicitly "none" vs invalid color */
                int is_none = 0;
                { const char *fv = fill_val; int fl = fill_vlen;
                  while (fl > 0 && isspace((unsigned char)*fv)) { fv++; fl--; }
                  while (fl > 0 && isspace((unsigned char)fv[fl-1])) fl--;
                  is_none = (fl == 4 && sigil__ci_prefix(fv, fl, "none"));
                }
                if (is_none) {
                    has_fill = 0; /* explicit none */
                } else {
                    has_fill = 1; /* invalid color: use initial value (black) */
                    fill_color[0] = 0; fill_color[1] = 0; fill_color[2] = 0; fill_color[3] = 1;
                }
            } else {
                has_fill = 1;
            }
        }

        /* Inherit fill from parent groups when not explicitly set */
        if (fill_vlen == 0 && !sigil__tag_is(&tag, "line")) {
            for (int gi = xform_depth; gi >= 0; gi--) {
                if (!g_style_stack[gi]) continue;
                const char *gf;
                int gflen = sigil__get_attr(g_style_stack[gi], g_style_len_stack[gi], "fill", &gf);
                if (gflen > 0) {
                    int gcr = sigil__parse_color(gf, gflen, fill_color);
                    if (gcr == SIGIL_COLOR_CURRENT) { memcpy(fill_color, current_color, 16); has_fill = 1; }
                    else if (gcr == SIGIL_COLOR_VALID) has_fill = 1;
                    else has_fill = 0;
                    break;
                }
            }
        }

        /* Stroke color with currentColor/inherit/gradient support */
        const char *stroke_val;
        int stroke_vlen = sigil__get_prop(tag.attrs, tag.attrs_len, style_str, style_len, "stroke", &stroke_val);
        if (stroke_vlen > 4 && memcmp(stroke_val, "url(", 4) == 0) {
            /* Stroke gradient */
            has_stroke = 0;
            const char *hash = (const char*)memchr(stroke_val, '#', (size_t)stroke_vlen);
            if (hash && grad_defs.count > 0) {
                hash++;
                const char *end_paren = (const char*)memchr(hash, ')', (size_t)(stroke_vlen - (int)(hash - stroke_val)));
                int sid_len = end_paren ? (int)(end_paren - hash) : (int)(stroke_vlen - (int)(hash - stroke_val));
                while (sid_len > 0 && isspace((unsigned char)hash[sid_len-1])) sid_len--;
                for (int gi = 0; gi < grad_defs.count; gi++) {
                    if ((int)strlen(grad_defs.data[gi].id) == sid_len &&
                        memcmp(grad_defs.data[gi].id, hash, (size_t)sid_len) == 0) {
                        if (grad_defs.data[gi].stop_count > 0 ||
                            grad_defs.data[gi].href[0] != '\0') {
                            stroke_gradient_idx = gi;
                            has_stroke = 1;
                        }
                        break;
                    }
                }
            }
            /* Fallback color after url() */
            if (!has_stroke) {
                const char *cp = (const char*)memchr(stroke_val, ')', (size_t)stroke_vlen);
                if (cp) {
                    cp++;
                    int rem = stroke_vlen - (int)(cp - stroke_val);
                    while (rem > 0 && isspace((unsigned char)*cp)) { cp++; rem--; }
                    if (rem > 0) {
                        int cr = sigil__parse_color(cp, rem, stroke_color);
                        if (cr == SIGIL_COLOR_CURRENT) { memcpy(stroke_color, current_color, 16); cr = 1; }
                        has_stroke = (cr == SIGIL_COLOR_VALID);
                    }
                }
            }
        } else if (stroke_vlen > 0) {
            int cr = sigil__parse_color(stroke_val, stroke_vlen, stroke_color);
            if (cr == SIGIL_COLOR_CURRENT) {
                memcpy(stroke_color, current_color, sizeof(float)*4); has_stroke = 1;
            } else if (cr == SIGIL_COLOR_INHERIT) {
                for (int gi = xform_depth; gi >= 0; gi--) {
                    if (!g_style_stack[gi]) continue;
                    const char *gs;
                    int gslen = sigil__get_attr(g_style_stack[gi], g_style_len_stack[gi], "stroke", &gs);
                    if (gslen > 0) { has_stroke = (sigil__parse_color(gs, gslen, stroke_color) == SIGIL_COLOR_VALID); break; }
                }
            } else {
                has_stroke = (cr == SIGIL_COLOR_VALID);
            }
        }
        /* Inherit stroke from parent groups when not explicitly set */
        if (stroke_vlen == 0) {
            for (int gi = xform_depth; gi >= 0; gi--) {
                if (!g_style_stack[gi]) continue;
                const char *gs;
                int gslen = sigil__get_attr(g_style_stack[gi], g_style_len_stack[gi], "stroke", &gs);
                if (gslen > 0) {
                    has_stroke = (sigil__parse_color(gs, gslen, stroke_color) == SIGIL_COLOR_VALID);
                    break;
                }
            }
        }

        {
            const char *sw_val;
            int sw_len = sigil__get_prop(tag.attrs, tag.attrs_len, style_str, style_len, "stroke-width", &sw_val);
            if (sw_len > 0) stroke_width = sigil__parse_length(sw_val, sw_len, vp_diag);
        }
        /* Inherit stroke-width from parent groups */
        if (stroke_width == 0 && has_stroke) {
            for (int gi = xform_depth; gi >= 0; gi--) {
                if (!g_style_stack[gi]) continue;
                float gsw = sigil__get_attr_float(g_style_stack[gi], g_style_len_stack[gi], "stroke-width", 0);
                if (gsw > 0) { stroke_width = gsw; break; }
            }
        }
        if (has_stroke && stroke_width == 0) stroke_width = 1.0f; /* SVG default */

        /* Parse opacity values — handle both number and percentage.
           Reject invalid values with unit suffixes (e.g. "0.1mm"). */
        #define SIGIL__PARSE_OPACITY(prop, var) do { \
            const char *_ov; \
            int _ol = sigil__get_prop(tag.attrs, tag.attrs_len, style_str, style_len, prop, &_ov); \
            if (_ol > 0) { \
                char *_ep; float _v = strtof(_ov, &_ep); \
                int _rem = _ol - (int)(_ep - _ov); \
                while (_rem > 0 && isspace((unsigned char)*_ep)) { _ep++; _rem--; } \
                if (_rem == 0) { var = _v; } \
                else if (_rem == 1 && *_ep == '%') { var = _v / 100.0f; } \
                /* else: invalid unit suffix, keep default */ \
                if (var < 0) var = 0; if (var > 1) var = 1; \
            } \
        } while(0)
        SIGIL__PARSE_OPACITY("opacity", opacity);
        SIGIL__PARSE_OPACITY("fill-opacity", fill_opacity);
        SIGIL__PARSE_OPACITY("stroke-opacity", stroke_opacity);
        #undef SIGIL__PARSE_OPACITY

        /* Multiply in parent group opacity values (opacity is NOT inherited per spec,
           but it does compose: each group's opacity multiplies into descendants) */
        {
            const char *_ov;
            int _ol = sigil__get_prop(tag.attrs, tag.attrs_len, style_str, style_len, "opacity", &_ov);
            if (_ol == 0) {
                /* No opacity on this element — check parent groups for opacity */
                for (int gi = xform_depth; gi >= 0; gi--) {
                    if (!g_style_stack[gi]) continue;
                    int gol = sigil__get_attr(g_style_stack[gi], g_style_len_stack[gi], "opacity", &_ov);
                    if (gol > 0) {
                        char *_ep; float gop = strtof(_ov, &_ep);
                        int _rem = gol - (int)(_ep - _ov);
                        while (_rem > 0 && isspace((unsigned char)*_ep)) { _ep++; _rem--; }
                        if (_rem == 0 || (_rem == 1 && *_ep == '%')) {
                            if (_rem == 1 && *_ep == '%') gop /= 100.0f;
                            if (gop < 0) gop = 0; if (gop > 1) gop = 1;
                            opacity *= gop;
                        }
                        break;
                    }
                }
            }
        }

        /* Inherit fill-opacity from parent groups */
        {
            const char *fo_val;
            int fo_len = sigil__get_prop(tag.attrs, tag.attrs_len, style_str, style_len, "fill-opacity", &fo_val);
            if (fo_len == 0) {
                for (int gi = xform_depth; gi >= 0; gi--) {
                    if (!g_style_stack[gi]) continue;
                    float gfo = sigil__get_attr_float(g_style_stack[gi], g_style_len_stack[gi], "fill-opacity", -1);
                    if (gfo >= 0) { fill_opacity = gfo; break; }
                }
            }
        }
        /* Inherit stroke-opacity from parent groups */
        {
            const char *so_val;
            int so_len = sigil__get_prop(tag.attrs, tag.attrs_len, style_str, style_len, "stroke-opacity", &so_val);
            if (so_len == 0) {
                for (int gi = xform_depth; gi >= 0; gi--) {
                    if (!g_style_stack[gi]) continue;
                    float gso = sigil__get_attr_float(g_style_stack[gi], g_style_len_stack[gi], "stroke-opacity", -1);
                    if (gso >= 0) { stroke_opacity = gso; break; }
                }
            }
        }

        /* fill-rule: check style, then attribute */
        {
            const char *fr_val;
            int fr_len = sigil__get_prop(tag.attrs, tag.attrs_len, style_str, style_len, "fill-rule", &fr_val);
            if (fr_len == 7 && memcmp(fr_val, "evenodd", 7) == 0)
                fill_rule = SIGIL_FILL_EVENODD;
        }

        /* stroke-linejoin: element, then group inheritance */
        SigilLineJoin line_join = SIGIL_JOIN_MITER;
        {
            const char *lj_val;
            int lj_len = sigil__get_prop(tag.attrs, tag.attrs_len, style_str, style_len, "stroke-linejoin", &lj_val);
            if (lj_len == 0) {
                for (int gi = xform_depth; gi >= 0 && lj_len == 0; gi--) {
                    if (g_style_stack[gi])
                        lj_len = sigil__get_attr(g_style_stack[gi], g_style_len_stack[gi], "stroke-linejoin", &lj_val);
                }
            }
            if (lj_len == 5 && memcmp(lj_val, "round", 5) == 0)
                line_join = SIGIL_JOIN_ROUND;
            else if (lj_len == 5 && memcmp(lj_val, "bevel", 5) == 0)
                line_join = SIGIL_JOIN_BEVEL;
        }

        /* stroke-linecap: element, then group inheritance */
        SigilLineCap line_cap = SIGIL_CAP_BUTT;
        {
            const char *lc_val;
            int lc_len = sigil__get_prop(tag.attrs, tag.attrs_len, style_str, style_len, "stroke-linecap", &lc_val);
            if (lc_len == 0) {
                for (int gi = xform_depth; gi >= 0 && lc_len == 0; gi--) {
                    if (g_style_stack[gi])
                        lc_len = sigil__get_attr(g_style_stack[gi], g_style_len_stack[gi], "stroke-linecap", &lc_val);
                }
            }
            if (lc_len == 5 && memcmp(lc_val, "round", 5) == 0)
                line_cap = SIGIL_CAP_ROUND;
            else if (lc_len == 6 && memcmp(lc_val, "square", 6) == 0)
                line_cap = SIGIL_CAP_SQUARE;
        }

        /* stroke-miterlimit: element, then group inheritance.
           miterlimit is a <number>, not a <length> — reject unit suffixes */
        float miter_limit = 4.0f;
        {
            const char *ml_val;
            int ml_len = sigil__get_prop(tag.attrs, tag.attrs_len, style_str, style_len, "stroke-miterlimit", &ml_val);
            if (ml_len == 0) {
                for (int gi = xform_depth; gi >= 0 && ml_len == 0; gi--) {
                    if (g_style_stack[gi])
                        ml_len = sigil__get_attr(g_style_stack[gi], g_style_len_stack[gi], "stroke-miterlimit", &ml_val);
                }
            }
            if (ml_len > 0) {
                char *ml_end;
                float ml_v = strtof(ml_val, &ml_end);
                /* Reject if has unit suffix (miterlimit is a pure number) */
                int ml_rem = ml_len - (int)(ml_end - ml_val);
                while (ml_rem > 0 && isspace((unsigned char)*ml_end)) { ml_end++; ml_rem--; }
                if (ml_rem == 0 && ml_v >= 1.0f)
                    miter_limit = ml_v;
                /* else: invalid value, keep default 4.0 */
            }
            if (miter_limit < 1.0f) miter_limit = 4.0f; /* SVG spec: values < 1 are invalid, use default */
        }

        /* stroke-dasharray: parse dash pattern */
        float dash_array[16];
        int dash_count = 0;
        float dash_offset = 0;
        {
            const char *da_val;
            int da_len = sigil__get_prop(tag.attrs, tag.attrs_len, style_str, style_len, "stroke-dasharray", &da_val);
            if (da_len == 0) {
                for (int gi = xform_depth; gi >= 0 && da_len == 0; gi--) {
                    if (g_style_stack[gi])
                        da_len = sigil__get_attr(g_style_stack[gi], g_style_len_stack[gi], "stroke-dasharray", &da_val);
                }
            }
            if (da_len > 0 && da_val && !(da_len == 4 && memcmp(da_val, "none", 4) == 0)) {
                const char *p = da_val;
                const char *end = da_val + da_len;
                while (p < end && dash_count < 16) {
                    while (p < end && (isspace((unsigned char)*p) || *p == ',')) p++;
                    if (p >= end) break;
                    float v = sigil__parse_length(p, (int)(end - p), vp_diag);
                    char *ep;
                    strtof(p, &ep);
                    if (ep == p) break;
                    /* Skip past number and any unit suffix */
                    p = ep;
                    while (p < end && isalpha((unsigned char)*p)) p++;
                    while (p < end && *p == '%') p++;
                    if (v < 0) v = 0;
                    dash_array[dash_count++] = v;
                }
                /* SVG spec: if odd count, repeat the array to make even */
                if (dash_count > 0 && (dash_count & 1)) {
                    int orig = dash_count;
                    for (int i = 0; i < orig && dash_count < 16; i++)
                        dash_array[dash_count++] = dash_array[i];
                }
                /* Check if all zeros (equivalent to none) */
                int all_zero = 1;
                for (int i = 0; i < dash_count; i++)
                    if (dash_array[i] > 0) { all_zero = 0; break; }
                if (all_zero) dash_count = 0;
            }
            /* stroke-dashoffset */
            const char *do_val;
            int do_len = sigil__get_prop(tag.attrs, tag.attrs_len, style_str, style_len, "stroke-dashoffset", &do_val);
            if (do_len == 0) {
                for (int gi = xform_depth; gi >= 0 && do_len == 0; gi--) {
                    if (g_style_stack[gi])
                        do_len = sigil__get_attr(g_style_stack[gi], g_style_len_stack[gi], "stroke-dashoffset", &do_val);
                }
            }
            if (do_len > 0) dash_offset = sigil__parse_length(do_val, do_len, vp_diag);
        }

        /* Get element transform (transform is not a CSS property, only an attribute) */
        float elem_xform[6];
        const char *etr;
        int etrlen = sigil__get_attr(tag.attrs, tag.attrs_len, "transform", &etr);
        if (etrlen > 0 && etr) {
            sigil__parse_transform(etr, etrlen, elem_xform);
        } else {
            sigil__mat_identity(elem_xform);
        }

        /* Compute accumulated transform: group_stack * element_transform */
        float accum[6];
        sigil__mat_multiply(xform_stack[xform_depth], elem_xform, accum);

        /* Apply transform to curves */
        int is_identity = (fabsf(accum[0] - 1.0f) < 1e-6f &&
                          fabsf(accum[1]) < 1e-6f &&
                          fabsf(accum[2]) < 1e-6f &&
                          fabsf(accum[3] - 1.0f) < 1e-6f &&
                          fabsf(accum[4]) < 1e-6f &&
                          fabsf(accum[5]) < 1e-6f);
        if (!is_identity && curve_count > 0) {
            sigil__transform_curves(curves, curve_count, accum, &shape_bounds);
        }

        /* Apply stroke dash array before stroke-to-fill */
        SigilCurve *dashed_curves = NULL;
        int dashed_count = 0;
        if (has_stroke && dash_count > 0 && curve_count > 0) {
            SigilBounds dash_bounds;
            dashed_count = sigil__apply_dash(curves, curve_count,
                                              dash_array, dash_count, dash_offset,
                                              &dashed_curves, &dash_bounds);
        }

        /* Implicitly close open subpaths for fill (SVG spec: open subpaths
           are treated as closed when filling) */
        if (has_fill && curve_count > 0) {
            float firstX = curves[0].p0x, firstY = curves[0].p0y;
            float lastX  = curves[curve_count - 1].p2x;
            float lastY  = curves[curve_count - 1].p2y;
            if (fabsf(lastX - firstX) > 1e-4f || fabsf(lastY - firstY) > 1e-4f) {
                curves = (SigilCurve *)realloc(curves,
                    (size_t)(curve_count + 1) * sizeof(SigilCurve));
                curves[curve_count] = sigil__line_to_quad(lastX, lastY, firstX, firstY);
                curve_count++;
            }
        }

        /* Handle stroke-to-fill conversion */
        if (!has_fill && has_stroke && stroke_width > 0 && curve_count > 0) {
            /* Fill=none, stroke set: convert stroke to fill */
            SigilCurve *stroke_src = (dashed_count > 0) ? dashed_curves : curves;
            int stroke_src_count = (dashed_count > 0) ? dashed_count : curve_count;
            SigilCurve *stroke_curves = NULL;
            SigilBounds stroke_bounds;
            /* Try smooth offset first (for circles, ellipses, etc.) */
            int sc = sigil__stroke_smooth_closed(stroke_src, stroke_src_count,
                                                  stroke_width,
                                                  &stroke_curves, &stroke_bounds);
            if (sc == 0) {
                /* Fall back to polyline-based stroke expansion */
                sc = sigil__stroke_to_fill(stroke_src, stroke_src_count, stroke_width,
                                            line_join, line_cap, miter_limit,
                                            &stroke_curves, &stroke_bounds);
            }
            free(curves);
            if (dashed_curves) { free(dashed_curves); dashed_curves = NULL; }
            curves = stroke_curves;
            curve_count = sc;
            shape_bounds = stroke_bounds;
            /* Use stroke color/gradient as fill */
            memcpy(fill_color, stroke_color, sizeof(float) * 4);
            fill_gradient_idx = stroke_gradient_idx;
            has_fill = 1;
            has_stroke = 0;
        }

        /* Create fill element */
        if (has_fill && curve_count > 0) {
            SigilElement *e = sigil__elem_array_push(&elems);
            e->curves = curves;
            e->curve_count = (uint32_t)curve_count;
            memcpy(e->fill_color, fill_color, sizeof(float) * 4);
            memcpy(e->stroke_color, stroke_color, sizeof(float) * 4);
            e->stroke_width = stroke_width;
            memcpy(e->transform, accum, sizeof(float) * 6);
            e->fill_rule = fill_rule;
            e->opacity = opacity * fill_opacity;
            e->bounds = shape_bounds;
            e->fill_gradient_idx = fill_gradient_idx;
            sigil__build_bands(e);
            curves = NULL; /* ownership transferred */
        }

        /* If both fill and stroke, add a second element for stroke outline */
        if (has_fill && has_stroke && stroke_width > 0 && curves == NULL &&
            elems.count > 0) {
            SigilElement *fillElem = &elems.data[elems.count - 1];
            SigilCurve *stroke_src2 = (dashed_count > 0) ? dashed_curves : fillElem->curves;
            int stroke_src2_count = (dashed_count > 0) ? dashed_count : (int)fillElem->curve_count;
            SigilCurve *stroke_curves = NULL;
            SigilBounds stroke_bounds;
            int sc = sigil__stroke_to_fill(stroke_src2, stroke_src2_count,
                                            stroke_width, line_join, line_cap, miter_limit,
                                            &stroke_curves, &stroke_bounds);
            if (sc > 0) {
                SigilElement *se = sigil__elem_array_push(&elems);
                se->curves = stroke_curves;
                se->curve_count = (uint32_t)sc;
                memcpy(se->fill_color, stroke_color, sizeof(float) * 4);
                se->stroke_width = stroke_width;
                memcpy(se->transform, accum, sizeof(float) * 6);
                se->fill_rule = fill_rule;
                se->opacity = opacity * stroke_opacity;
                se->bounds = stroke_bounds;
                se->fill_gradient_idx = stroke_gradient_idx;
                sigil__build_bands(se);
            }
        }

        /* Free curves if not transferred */
        free(curves);
        free(dashed_curves);

        /* Pop the use element's transform frame after rendering the shape */
        if (use_shape_frame && xform_depth > 0) xform_depth--;

        /* Skip children of basic shape elements (they can't have child shapes per SVG spec) */
        if (is_shape && !tag.self_close) {
            /* Find the matching closing tag for this basic shape */
            const char *tn = tag.name;
            int tnl = tag.name_len;
            int depth = 1;
            SigilTag skip_tag;
            while (depth > 0 && sigil__next_tag(svg_data, (int)len, &pos, &skip_tag)) {
                if (skip_tag.name_len == tnl && memcmp(skip_tag.name, tn, (size_t)tnl) == 0) {
                    if (skip_tag.is_close) depth--;
                    else if (!skip_tag.self_close) depth++;
                }
            }
        }
    }

    scene->elements = elems.data;
    scene->element_count = elems.count;

    /* Resolve gradient href inheritance and store in scene */
    if (grad_defs.count > 0) {
        sigil__resolve_gradient_hrefs(&grad_defs);
        /* Post-pass: fix radial gradient focal defaults (fx/fy default to cx/cy)
           and enforce monotonically increasing stop offsets */
        for (int i = 0; i < grad_defs.count; i++) {
            SigilGradientDef *g = &grad_defs.data[i];
            if (g->type == 2) {
                if (g->fx < -0.5f) g->fx = g->cx;
                if (g->fy < -0.5f) g->fy = g->cy;
            }
            /* SVG spec: stop offsets must be monotonically non-decreasing */
            for (int s = 1; s < g->stop_count; s++) {
                if (g->stops[s].offset < g->stops[s-1].offset)
                    g->stops[s].offset = g->stops[s-1].offset;
            }
            /* userSpaceOnUse: convert default (0-1) and percent values to viewport coords.
               Default values are stored as objectBoundingBox fractions (0-1 range).
               For userSpaceOnUse, SVG spec defines defaults as percentages of the viewport. */
            if (!g->objectBBox) {
                float vw = scene->viewBox[2] > 0 ? scene->viewBox[2] : (scene->width > 0 ? scene->width : 300.0f);
                float vh = scene->viewBox[3] > 0 ? scene->viewBox[3] : (scene->height > 0 ? scene->height : 150.0f);
                if (g->type == 1) {
                    /* Linear: defaults x1=0%, y1=0%, x2=100%, y2=0% are stored as 0,0,1,0 */
                    if (!(g->attrs_set & SIGIL_GRAD_HAS_X1) || (g->attrs_set & SIGIL_GRAD_PCT_X1)) g->x1 *= vw;
                    if (!(g->attrs_set & SIGIL_GRAD_HAS_Y1) || (g->attrs_set & SIGIL_GRAD_PCT_Y1)) g->y1 *= vh;
                    if (!(g->attrs_set & SIGIL_GRAD_HAS_X2) || (g->attrs_set & SIGIL_GRAD_PCT_X2)) g->x2 *= vw;
                    if (!(g->attrs_set & SIGIL_GRAD_HAS_Y2) || (g->attrs_set & SIGIL_GRAD_PCT_Y2)) g->y2 *= vh;
                } else if (g->type == 2) {
                    /* Radial: defaults cx=50%, cy=50%, r=50% stored as 0.5, 0.5, 0.5 */
                    if (!(g->attrs_set & SIGIL_GRAD_HAS_CX) || (g->attrs_set & SIGIL_GRAD_PCT_CX)) g->cx *= vw;
                    if (!(g->attrs_set & SIGIL_GRAD_HAS_CY) || (g->attrs_set & SIGIL_GRAD_PCT_CY)) g->cy *= vh;
                    float diag = sqrtf(vw*vw + vh*vh) / 1.41421356f; /* normalized diagonal */
                    /* r percentage is relative to the normalized diagonal per SVG 2 spec */
                    if (!(g->attrs_set & SIGIL_GRAD_HAS_R)  || (g->attrs_set & SIGIL_GRAD_PCT_R))  g->r  *= diag;
                    if (g->attrs_set & SIGIL_GRAD_PCT_FX) g->fx *= vw;
                    if (g->attrs_set & SIGIL_GRAD_PCT_FY) g->fy *= vh;
                    if (!(g->attrs_set & SIGIL_GRAD_HAS_FR) || (g->attrs_set & SIGIL_GRAD_PCT_FR)) g->fr *= diag;
                    /* Re-apply focal defaults after coordinate conversion */
                    if (g->fx < -0.5f) g->fx = g->cx;
                    if (g->fy < -0.5f) g->fy = g->cy;
                }
            }
        }
        scene->gradients = grad_defs.data;
        scene->gradient_count = grad_defs.count;
    }

    return scene;
}

/* ------------------------------------------------------------------ */
/*  Band sorting helper                                               */
/* ------------------------------------------------------------------ */

typedef struct { int idx; float key; } SigilSortEntry;

static int sigil__cmp_desc(const void *a, const void *b) {
    float ka = ((const SigilSortEntry *)a)->key;
    float kb = ((const SigilSortEntry *)b)->key;
    return (ka < kb) - (ka > kb);
}

/* axis 0 = sort by max-x (horizontal bands),  axis 1 = sort by max-y (vertical bands) */
static void sigil__sort_band(SigilBandEntry *band, SigilCurve *curves, int axis) {
    int n = band->count;
    if (n <= 1) return;
    SigilSortEntry *e = (SigilSortEntry *)malloc((size_t)n * sizeof *e);
    for (int i = 0; i < n; i++) {
        SigilCurve *c = &curves[band->curveIndices[i]];
        float mx = axis == 0
            ? fmaxf(fmaxf(c->p0x, c->p1x), c->p2x)
            : fmaxf(fmaxf(c->p0y, c->p1y), c->p2y);
        SigilSortEntry entry = {band->curveIndices[i], mx};
        e[i] = entry;
    }
    qsort(e, (size_t)n, sizeof *e, sigil__cmp_desc);
    for (int i = 0; i < n; i++) band->curveIndices[i] = e[i].idx;
    free(e);
}

/* ------------------------------------------------------------------ */
/*  Shader file loading helper                                        */
/* ------------------------------------------------------------------ */

static char* sigil__read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)sz + 1);
    size_t rd = fread(buf, 1, (size_t)sz, f);
    buf[rd] = '\0';
    fclose(f);
    return buf;
}

#ifndef STRVIEW
#ifdef __cplusplus
#define STRVIEW(X) WGPUStringView{X, sizeof(X) - 1}
#else
#define STRVIEW(X) (WGPUStringView){X, sizeof(X) - 1}
#endif
#endif

/* ------------------------------------------------------------------ */
/*  sigil_create — GPU pipeline setup                                 */
/* ------------------------------------------------------------------ */

SigilContext* sigil_create(WGPUDevice device, WGPUTextureFormat colorFormat,
                           WGPUTextureFormat depthFormat) {
    SigilContext *ctx = (SigilContext *)calloc(1, sizeof(SigilContext));
    if (!ctx) return NULL;
    ctx->device = device;
    ctx->queue = wgpuDeviceGetQueue(device);
    ctx->colorFormat = colorFormat;
    ctx->depthFormat = depthFormat;

    /* Load shaders — check SIGIL_SHADER_PATH env var, else current directory,
       then fall back to parent directory */
    const char *shaderDir = getenv("SIGIL_SHADER_PATH");
    char vsPath[1024], fsPath[1024];
    char *vsSrc = NULL, *fsSrc = NULL;
    if (shaderDir && shaderDir[0]) {
        snprintf(vsPath, sizeof vsPath, "%s/sigil_vertex.wgsl", shaderDir);
        snprintf(fsPath, sizeof fsPath, "%s/sigil_fragment.wgsl", shaderDir);
        vsSrc = sigil__read_file(vsPath);
        fsSrc = sigil__read_file(fsPath);
    }
    if (!vsSrc || !fsSrc) {
        free(vsSrc); free(fsSrc);
        snprintf(vsPath, sizeof vsPath, "sigil_vertex.wgsl");
        snprintf(fsPath, sizeof fsPath, "sigil_fragment.wgsl");
        vsSrc = sigil__read_file(vsPath);
        fsSrc = sigil__read_file(fsPath);
    }
    if (!vsSrc || !fsSrc) {
        free(vsSrc); free(fsSrc);
        snprintf(vsPath, sizeof vsPath, "../sigil_vertex.wgsl");
        snprintf(fsPath, sizeof fsPath, "../sigil_fragment.wgsl");
        vsSrc = sigil__read_file(vsPath);
        fsSrc = sigil__read_file(fsPath);
    }
    if (!vsSrc || !fsSrc) {
        fprintf(stderr, "sigil_create: shader files not found (searched CWD, ../, SIGIL_SHADER_PATH)\n");
        free(vsSrc); free(fsSrc); free(ctx);
        return NULL;
    }

    /* Create shader modules */
    WGPUShaderSourceWGSL vsWgsl = {
        .chain = {.sType = WGPUSType_ShaderSourceWGSL, .next = NULL},
        .code = {.data = vsSrc, .length = WGPU_STRLEN}
    };
    WGPUShaderModuleDescriptor vsModDesc = {.nextInChain = &vsWgsl.chain, .label = {NULL, WGPU_STRLEN}};
    ctx->vertexShader = wgpuDeviceCreateShaderModule(device, &vsModDesc);

    WGPUShaderSourceWGSL fsWgsl = {
        .chain = {.sType = WGPUSType_ShaderSourceWGSL, .next = NULL},
        .code = {.data = fsSrc, .length = WGPU_STRLEN}
    };
    WGPUShaderModuleDescriptor fsModDesc = {.nextInChain = &fsWgsl.chain, .label = {NULL, WGPU_STRLEN}};
    ctx->fragmentShader = wgpuDeviceCreateShaderModule(device, &fsModDesc);

    free(vsSrc); free(fsSrc);

    /* Bind group layout: uniform + curve SSBO + band SSBO + gradient texture + sampler */
    WGPUBindGroupLayoutEntry bglEntries[5] = {
        { .binding = 0, .visibility = WGPUShaderStage_Vertex,
          .buffer = {.type = WGPUBufferBindingType_Uniform, .nextInChain = NULL, .hasDynamicOffset = 0} , .nextInChain = NULL, .bindingArraySize = 0},
        { .binding = 1, .visibility = WGPUShaderStage_Fragment,
          .buffer = {.type = WGPUBufferBindingType_ReadOnlyStorage, .nextInChain = NULL, .hasDynamicOffset = 0} , .nextInChain = NULL, .bindingArraySize = 0},
        { .binding = 2, .visibility = WGPUShaderStage_Fragment,
          .buffer = {.type = WGPUBufferBindingType_ReadOnlyStorage, .nextInChain = NULL, .hasDynamicOffset = 0} , .nextInChain = NULL, .bindingArraySize = 0},
        { .binding = 3, .visibility = WGPUShaderStage_Fragment,
          .texture = {.sampleType = WGPUTextureSampleType_Float,
                      .viewDimension = WGPUTextureViewDimension_2D, .nextInChain = NULL, .multisampled = 0} , .nextInChain = NULL, .bindingArraySize = 0},
        { .binding = 4, .visibility = WGPUShaderStage_Fragment,
          .sampler = {.type = WGPUSamplerBindingType_Filtering, .nextInChain = NULL} , .nextInChain = NULL, .bindingArraySize = 0},
    };
    WGPUBindGroupLayoutDescriptor renderBGLDesc = {.entryCount = 5, .entries = bglEntries, .nextInChain = NULL, .label = {NULL, WGPU_STRLEN}};
    ctx->renderBGL = wgpuDeviceCreateBindGroupLayout(device, &renderBGLDesc);
    WGPUPipelineLayoutDescriptor renderPLDesc = {.bindGroupLayoutCount = 1,
        .bindGroupLayouts = &ctx->renderBGL, .nextInChain = NULL, .label = {NULL, WGPU_STRLEN}};
    ctx->renderPipelineLayout = wgpuDeviceCreatePipelineLayout(device, &renderPLDesc);

    /* Vertex attributes: 7 x Float32x4, stride = 112 */
    WGPUVertexAttribute vAttrs[7] = {
        {.shaderLocation = 0, .offset =  0, .format = WGPUVertexFormat_Float32x4, .nextInChain = NULL},
        {.shaderLocation = 1, .offset = 16, .format = WGPUVertexFormat_Float32x4, .nextInChain = NULL},
        {.shaderLocation = 2, .offset = 32, .format = WGPUVertexFormat_Float32x4, .nextInChain = NULL},
        {.shaderLocation = 3, .offset = 48, .format = WGPUVertexFormat_Float32x4, .nextInChain = NULL},
        {.shaderLocation = 4, .offset = 64, .format = WGPUVertexFormat_Float32x4, .nextInChain = NULL},
        {.shaderLocation = 5, .offset = 80, .format = WGPUVertexFormat_Float32x4, .nextInChain = NULL},
        {.shaderLocation = 6, .offset = 96, .format = WGPUVertexFormat_Float32x4, .nextInChain = NULL},
    };
    WGPUVertexBufferLayout vbLayout = {
        .arrayStride = 112, .attributeCount = 7,
        .attributes = vAttrs, .stepMode = WGPUVertexStepMode_Vertex
    , .nextInChain = NULL};

    /* Premultiplied alpha blending */
    WGPUBlendState blend = {
        .color = {.srcFactor = WGPUBlendFactor_One,
                  .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,
                  .operation = WGPUBlendOperation_Add},
        .alpha = {.srcFactor = WGPUBlendFactor_One,
                  .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,
                  .operation = WGPUBlendOperation_Add},
    };
    WGPUColorTargetState cts = {
        .format = colorFormat,
        .blend = &blend,
        .writeMask = WGPUColorWriteMask_All
    , .nextInChain = NULL};
    WGPUFragmentState fs = {
        .module = ctx->fragmentShader, .entryPoint = STRVIEW("main"),
        .targetCount = 1, .targets = &cts
    , .nextInChain = NULL, .constantCount = 0, .constants = NULL};

    WGPURenderPipelineDescriptor rpDesc = {
        .layout = ctx->renderPipelineLayout,
        .vertex = {
            .module = ctx->vertexShader, .entryPoint = STRVIEW("main"),
            .bufferCount = 1, .buffers = &vbLayout
        , .nextInChain = NULL, .constantCount = 0, .constants = NULL},
        .fragment = &fs,
        .primitive = {
            .topology = WGPUPrimitiveTopology_TriangleList,
            .cullMode = WGPUCullMode_None
        , .nextInChain = NULL},
        .multisample = {.count = 1, .mask = 0xFFFFFFFF, .nextInChain = NULL},
     .nextInChain = NULL, .label = {NULL, WGPU_STRLEN}};

    /* Optional depth stencil */
    WGPUDepthStencilState dsState;
    if (depthFormat != 0) {
        memset(&dsState, 0, sizeof dsState);
        dsState.format = depthFormat;
        dsState.depthWriteEnabled = WGPUOptionalBool_True;
        dsState.depthCompare = WGPUCompareFunction_LessEqual;
        rpDesc.depthStencil = &dsState;
    }

    ctx->pipeline = wgpuDeviceCreateRenderPipeline(device, &rpDesc);

    /* ---- Gradient sampler (shared by render + gradient compute) ---- */
    WGPUSamplerDescriptor gradSamplerDesc = {
        .magFilter = WGPUFilterMode_Linear,
        .minFilter = WGPUFilterMode_Linear,
        .addressModeU = WGPUAddressMode_ClampToEdge,
        .addressModeV = WGPUAddressMode_ClampToEdge,
        .addressModeW = WGPUAddressMode_ClampToEdge,
        .nextInChain = NULL, .label = {NULL, WGPU_STRLEN}};
    ctx->gradientSampler = wgpuDeviceCreateSampler(device, &gradSamplerDesc);

    /* ---- Prepare compute: bind group layouts ---- */
    /* group(0): scene data inputs (read-only) */
    WGPUBindGroupLayoutEntry prepInEntries[5] = {
        { .binding = 0, .visibility = WGPUShaderStage_Compute,
          .buffer = {.type = WGPUBufferBindingType_ReadOnlyStorage, .nextInChain = NULL, .hasDynamicOffset = 0} , .nextInChain = NULL, .bindingArraySize = 0},
        { .binding = 1, .visibility = WGPUShaderStage_Compute,
          .buffer = {.type = WGPUBufferBindingType_ReadOnlyStorage, .nextInChain = NULL, .hasDynamicOffset = 0} , .nextInChain = NULL, .bindingArraySize = 0},
        { .binding = 2, .visibility = WGPUShaderStage_Compute,
          .buffer = {.type = WGPUBufferBindingType_ReadOnlyStorage, .nextInChain = NULL, .hasDynamicOffset = 0} , .nextInChain = NULL, .bindingArraySize = 0},
        { .binding = 3, .visibility = WGPUShaderStage_Compute,
          .buffer = {.type = WGPUBufferBindingType_ReadOnlyStorage, .nextInChain = NULL, .hasDynamicOffset = 0} , .nextInChain = NULL, .bindingArraySize = 0},
        { .binding = 4, .visibility = WGPUShaderStage_Compute,
          .buffer = {.type = WGPUBufferBindingType_Uniform, .nextInChain = NULL, .hasDynamicOffset = 0} , .nextInChain = NULL, .bindingArraySize = 0},
    };
    WGPUBindGroupLayoutDescriptor prepInBGLDesc = {.entryCount = 5, .entries = prepInEntries, .nextInChain = NULL, .label = {NULL, WGPU_STRLEN}};
    ctx->prepareInputBGL = wgpuDeviceCreateBindGroupLayout(device, &prepInBGLDesc);

    /* group(1): output buffers (read-write) */
    WGPUBindGroupLayoutEntry prepOutEntries[4] = {
        { .binding = 0, .visibility = WGPUShaderStage_Compute,
          .buffer = {.type = WGPUBufferBindingType_Storage, .nextInChain = NULL, .hasDynamicOffset = 0} , .nextInChain = NULL, .bindingArraySize = 0},
        { .binding = 1, .visibility = WGPUShaderStage_Compute,
          .buffer = {.type = WGPUBufferBindingType_Storage, .nextInChain = NULL, .hasDynamicOffset = 0} , .nextInChain = NULL, .bindingArraySize = 0},
        { .binding = 2, .visibility = WGPUShaderStage_Compute,
          .buffer = {.type = WGPUBufferBindingType_Storage, .nextInChain = NULL, .hasDynamicOffset = 0} , .nextInChain = NULL, .bindingArraySize = 0},
        { .binding = 3, .visibility = WGPUShaderStage_Compute,
          .buffer = {.type = WGPUBufferBindingType_Storage, .nextInChain = NULL, .hasDynamicOffset = 0} , .nextInChain = NULL, .bindingArraySize = 0},
    };
    WGPUBindGroupLayoutDescriptor prepOutBGLDesc = {.entryCount = 4, .entries = prepOutEntries, .nextInChain = NULL, .label = {NULL, WGPU_STRLEN}};
    ctx->prepareOutputBGL = wgpuDeviceCreateBindGroupLayout(device, &prepOutBGLDesc);

    WGPUBindGroupLayout prepareBGLs[2] = { ctx->prepareInputBGL, ctx->prepareOutputBGL };
    WGPUPipelineLayoutDescriptor preparePLDesc = {.bindGroupLayoutCount = 2,
        .bindGroupLayouts = prepareBGLs, .nextInChain = NULL, .label = {NULL, WGPU_STRLEN}};
    ctx->preparePipelineLayout = wgpuDeviceCreatePipelineLayout(device, &preparePLDesc);

    /* ---- Gradient compute: bind group layout ---- */
    WGPUBindGroupLayoutEntry gradEntries[3] = {
        { .binding = 0, .visibility = WGPUShaderStage_Compute,
          .buffer = {.type = WGPUBufferBindingType_ReadOnlyStorage, .nextInChain = NULL, .hasDynamicOffset = 0} , .nextInChain = NULL, .bindingArraySize = 0},
        { .binding = 1, .visibility = WGPUShaderStage_Compute,
          .buffer = {.type = WGPUBufferBindingType_ReadOnlyStorage, .nextInChain = NULL, .hasDynamicOffset = 0} , .nextInChain = NULL, .bindingArraySize = 0},
        { .binding = 2, .visibility = WGPUShaderStage_Compute,
          .buffer = {.type = WGPUBufferBindingType_Storage, .nextInChain = NULL, .hasDynamicOffset = 0} , .nextInChain = NULL, .bindingArraySize = 0},
    };
    WGPUBindGroupLayoutDescriptor gradBGLDesc = {.entryCount = 3, .entries = gradEntries, .nextInChain = NULL, .label = {NULL, WGPU_STRLEN}};
    ctx->gradientBGL = wgpuDeviceCreateBindGroupLayout(device, &gradBGLDesc);
    WGPUPipelineLayoutDescriptor gradPLDesc = {.bindGroupLayoutCount = 1,
        .bindGroupLayouts = &ctx->gradientBGL, .nextInChain = NULL, .label = {NULL, WGPU_STRLEN}};
    ctx->gradientPipelineLayout = wgpuDeviceCreatePipelineLayout(device, &gradPLDesc);

    /* ---- Load compute shaders ---- */
    char csPreparePath[1024], csGradPath[1024];
    char *csPrepareSrc = NULL, *csGradSrc = NULL;
    if (shaderDir && shaderDir[0]) {
        snprintf(csPreparePath, sizeof csPreparePath, "%s/sigil_prepare_compute.wgsl", shaderDir);
        snprintf(csGradPath, sizeof csGradPath, "%s/sigil_bake_gradient.wgsl", shaderDir);
        csPrepareSrc = sigil__read_file(csPreparePath);
        csGradSrc    = sigil__read_file(csGradPath);
    }
    if (!csPrepareSrc || !csGradSrc) {
        free(csPrepareSrc); free(csGradSrc);
        snprintf(csPreparePath, sizeof csPreparePath, "sigil_prepare_compute.wgsl");
        snprintf(csGradPath, sizeof csGradPath, "sigil_bake_gradient.wgsl");
        csPrepareSrc = sigil__read_file(csPreparePath);
        csGradSrc    = sigil__read_file(csGradPath);
    }
    if (!csPrepareSrc || !csGradSrc) {
        free(csPrepareSrc); free(csGradSrc);
        snprintf(csPreparePath, sizeof csPreparePath, "../sigil_prepare_compute.wgsl");
        snprintf(csGradPath, sizeof csGradPath, "../sigil_bake_gradient.wgsl");
        csPrepareSrc = sigil__read_file(csPreparePath);
        csGradSrc    = sigil__read_file(csGradPath);
    }
    if (!csPrepareSrc || !csGradSrc) {
        fprintf(stderr, "sigil_create: compute shader files not found (searched CWD, ../, SIGIL_SHADER_PATH)\n");
        free(csPrepareSrc); free(csGradSrc);
        /* Non-fatal for now: leave compute pipelines NULL */
    } else {
        WGPUShaderSourceWGSL csPreWgsl = {
            .chain = {.sType = WGPUSType_ShaderSourceWGSL, .next = NULL},
            .code = {.data = csPrepareSrc, .length = WGPU_STRLEN}
        };
        WGPUShaderModuleDescriptor csPreModDesc = {.nextInChain = &csPreWgsl.chain, .label = {NULL, WGPU_STRLEN}};
        ctx->prepareShader = wgpuDeviceCreateShaderModule(device, &csPreModDesc);

        WGPUShaderSourceWGSL csGradWgsl = {
            .chain = {.sType = WGPUSType_ShaderSourceWGSL, .next = NULL},
            .code = {.data = csGradSrc, .length = WGPU_STRLEN}
        };
        WGPUShaderModuleDescriptor csGradModDesc = {.nextInChain = &csGradWgsl.chain, .label = {NULL, WGPU_STRLEN}};
        ctx->gradientShader = wgpuDeviceCreateShaderModule(device, &csGradModDesc);

        free(csPrepareSrc); free(csGradSrc);

        /* Create compute pipelines */
        WGPUComputePipelineDescriptor prepCPDesc = {
            .layout = ctx->preparePipelineLayout,
            .compute = {.module = ctx->prepareShader, .entryPoint = STRVIEW("main"), .nextInChain = NULL, .constantCount = 0, .constants = NULL},
            .nextInChain = NULL, .label = {NULL, WGPU_STRLEN}};
        ctx->preparePipeline = wgpuDeviceCreateComputePipeline(device, &prepCPDesc);
        WGPUComputePipelineDescriptor gradCPDesc = {
            .layout = ctx->gradientPipelineLayout,
            .compute = {.module = ctx->gradientShader, .entryPoint = STRVIEW("main"), .nextInChain = NULL, .constantCount = 0, .constants = NULL},
            .nextInChain = NULL, .label = {NULL, WGPU_STRLEN}};
        ctx->gradientPipeline = wgpuDeviceCreateComputePipeline(device, &gradCPDesc);
    }

    return ctx;
}

/* ------------------------------------------------------------------ */
/*  sigil_upload — flatten scene into GPU storage buffers             */
/* ------------------------------------------------------------------ */

SigilGPUScene* sigil_upload(SigilContext* ctx, SigilScene* scene) {
    if (!ctx || !scene || scene->element_count == 0) return NULL;

    uint32_t ec = (uint32_t)scene->element_count;
    SigilElement *elems = scene->elements;
    int gradCount = scene->gradient_count;
    int gradTexH = gradCount > 0 ? gradCount : 1;

    /* ---- Count total curves ---- */
    uint32_t totalCurves = 0;
    for (uint32_t ei = 0; ei < ec; ei++)
        totalCurves += elems[ei].curve_count;

    /* ---- Flatten curves buffer: 6 floats per curve ---- */
    size_t curvesBytes = (size_t)totalCurves * 6 * sizeof(float);
    if (curvesBytes < 4) curvesBytes = 4;
    float *curvesData = (float *)malloc(curvesBytes);
    uint32_t ci = 0;
    for (uint32_t ei = 0; ei < ec; ei++) {
        for (uint32_t k = 0; k < elems[ei].curve_count; k++) {
            SigilCurve *c = &elems[ei].curves[k];
            curvesData[ci * 6 + 0] = c->p0x;
            curvesData[ci * 6 + 1] = c->p0y;
            curvesData[ci * 6 + 2] = c->p1x;
            curvesData[ci * 6 + 3] = c->p1y;
            curvesData[ci * 6 + 4] = c->p2x;
            curvesData[ci * 6 + 5] = c->p2y;
            ci++;
        }
    }

    /* ---- Elements buffer: 22 u32/f32 per element (88 bytes) ---- */
    size_t elemBytes = (size_t)ec * 22 * sizeof(uint32_t);
    if (elemBytes < 4) elemBytes = 4;
    uint32_t *elemData = (uint32_t *)calloc(ec, 22 * sizeof(uint32_t));
    uint32_t curveOff = 0;
    for (uint32_t ei = 0; ei < ec; ei++) {
        SigilElement *e = &elems[ei];
        uint32_t *row = &elemData[ei * 22];

        row[0] = curveOff;               /* curve_offset */
        row[1] = e->curve_count;         /* curve_count */

        float xMin = e->bounds.xMin, yMin = e->bounds.yMin;
        float xMax = e->bounds.xMax, yMax = e->bounds.yMax;
        float ew = xMax - xMin, eh = yMax - yMin;

        memcpy(&row[2], &xMin, 4);       /* bounds_xMin */
        memcpy(&row[3], &yMin, 4);       /* bounds_yMin */
        memcpy(&row[4], &xMax, 4);       /* bounds_xMax */
        memcpy(&row[5], &yMax, 4);       /* bounds_yMax */

        /* fill color: RGBA 0-1, NOT premultiplied */
        memcpy(&row[6],  &e->fill_color[0], 4);
        memcpy(&row[7],  &e->fill_color[1], 4);
        memcpy(&row[8],  &e->fill_color[2], 4);
        memcpy(&row[9],  &e->fill_color[3], 4);

        row[10] = (uint32_t)e->fill_rule; /* fill_rule */
        int32_t gi = (int32_t)e->fill_gradient_idx;
        memcpy(&row[11], &gi, 4);         /* gradient_idx (i32) */
        memcpy(&row[12], &e->opacity, 4); /* opacity */
        row[13] = 0;                       /* pad */

        /* Gradient params: pre-compute grad0, grad1 */
        float grad0[4] = {0, 0, 0, 0};
        float grad1[4] = {0, 0, 0, 0};

        if (e->fill_gradient_idx >= 0 && e->fill_gradient_idx < gradCount) {
            const SigilGradientDef *gd = &scene->gradients[e->fill_gradient_idx];
            /* grad1.x = normalized Y in gradient texture (center of row) */
            grad1[0] = ((float)e->fill_gradient_idx + 0.5f) / (float)gradTexH;
            grad1[1] = (float)gd->type; /* 1=linear, 2=radial */
            grad1[3] = (float)gd->spread; /* 0=pad, 1=reflect, 2=repeat */

            if (gd->type == 1) {
                /* Linear gradient: transform endpoints to em-space */
                float gx1 = gd->x1, gy1 = gd->y1, gx2 = gd->x2, gy2 = gd->y2;
                if (gd->objectBBox) {
                    /* objectBoundingBox: 0-1 range maps to element bounds */
                    gx1 = xMin + gx1 * ew; gy1 = yMin + gy1 * eh;
                    gx2 = xMin + gx2 * ew; gy2 = yMin + gy2 * eh;
                }
                /* Apply gradientTransform */
                float tx1 = gd->transform[0]*gx1 + gd->transform[2]*gy1 + gd->transform[4];
                float ty1 = gd->transform[1]*gx1 + gd->transform[3]*gy1 + gd->transform[5];
                float tx2 = gd->transform[0]*gx2 + gd->transform[2]*gy2 + gd->transform[4];
                float ty2 = gd->transform[1]*gx2 + gd->transform[3]*gy2 + gd->transform[5];
                grad0[0] = tx1; grad0[1] = ty1; grad0[2] = tx2; grad0[3] = ty2;
            } else {
                /* Radial gradient: transform center/focal to em-space */
                float gcx = gd->cx, gcy = gd->cy, gfx = gd->fx, gfy = gd->fy;
                float gr = gd->r;
                if (gd->objectBBox) {
                    gcx = xMin + gcx * ew; gcy = yMin + gcy * eh;
                    gfx = xMin + gfx * ew; gfy = yMin + gfy * eh;
                    gr *= fmaxf(ew, eh); /* approximate: scale radius by max dimension */
                }
                /* Apply gradientTransform to center */
                float tcx = gd->transform[0]*gcx + gd->transform[2]*gcy + gd->transform[4];
                float tcy = gd->transform[1]*gcx + gd->transform[3]*gcy + gd->transform[5];
                float tfx = gd->transform[0]*gfx + gd->transform[2]*gfy + gd->transform[4];
                float tfy = gd->transform[1]*gfx + gd->transform[3]*gfy + gd->transform[5];
                /* Transform radius (use average scale factor) */
                float sx = sqrtf(gd->transform[0]*gd->transform[0] + gd->transform[1]*gd->transform[1]);
                float sy = sqrtf(gd->transform[2]*gd->transform[2] + gd->transform[3]*gd->transform[3]);
                float tscale = (sx + sy) * 0.5f;
                grad0[0] = tcx; grad0[1] = tcy; grad0[2] = tfx; grad0[3] = tfy;
                float tr = gr * tscale;  /* transformed outer radius */
                float tfr = gd->fr;
                if (gd->objectBBox) tfr *= fmaxf(ew, eh);
                tfr *= tscale;
                /* Encode fr/r ratio in fractional part of spread:
                   grad1[3] = spread + fr_ratio * 0.001
                   Shader decodes: fr_ratio = fract(val) * 1000 */
                float fr_ratio = (tr > 0) ? (tfr / tr) : 0.0f;
                if (fr_ratio < 0) fr_ratio = 0;
                if (fr_ratio > 99.0f) fr_ratio = 99.0f;
                grad1[2] = tr;
                grad1[3] = (float)gd->spread + fr_ratio * 0.001f;
            }
        }

        memcpy(&row[14], grad0, 16);  /* grad0: 4 floats */
        memcpy(&row[18], grad1, 16);  /* grad1: 4 floats */

        curveOff += e->curve_count;
    }

    /* ---- Offsets buffer: per-element {curve_start, band_start} ---- */
    uint32_t *offsetData = (uint32_t *)malloc((size_t)ec * 2 * sizeof(uint32_t));
    uint32_t curveStart = 0, bandStart = 0;
    for (uint32_t ei = 0; ei < ec; ei++) {
        offsetData[ei * 2 + 0] = curveStart;
        offsetData[ei * 2 + 1] = bandStart;
        curveStart += elems[ei].curve_count * 2; /* 2 vec4s per curve */
        bandStart  += 16 + elems[ei].curve_count * 16; /* worst-case band alloc */
    }
    /* curveStart and bandStart are now the final totals (output buffer sizes) */
    uint32_t curveOutVec4s = curveStart;
    uint32_t bandOutVec4s  = bandStart;
    size_t offsetsBytes = (size_t)ec * 2 * sizeof(uint32_t);
    if (offsetsBytes < 4) offsetsBytes = 4;

    /* ---- Gradient buffers ---- */
    uint32_t *gradBufData = NULL;
    float *stopBufData = NULL;
    size_t gradBufBytes = 4; /* minimum non-zero */
    size_t stopBufBytes = 4;
    uint32_t totalStops = 0;

    if (gradCount > 0) {
        /* 24 u32/f32 per gradient (96 bytes) */
        gradBufBytes = (size_t)gradCount * 24 * sizeof(uint32_t);
        gradBufData = (uint32_t *)calloc((size_t)gradCount, 24 * sizeof(uint32_t));

        /* Count total stops for sizing */
        for (int gi = 0; gi < gradCount; gi++)
            totalStops += (uint32_t)scene->gradients[gi].stop_count;

        /* 8 floats per stop (32 bytes) */
        stopBufBytes = totalStops > 0 ? (size_t)totalStops * 8 * sizeof(float) : 4;
        stopBufData = (float *)calloc(totalStops > 0 ? totalStops : 1, 8 * sizeof(float));

        uint32_t stopOffset = 0;
        for (int gi = 0; gi < gradCount; gi++) {
            const SigilGradientDef *gd = &scene->gradients[gi];
            uint32_t *gRow = &gradBufData[gi * 24];

            /* First 8 words: i32 fields */
            int32_t typeVal = (int32_t)gd->type;
            int32_t spreadVal = (int32_t)gd->spread;
            int32_t scVal = (int32_t)gd->stop_count;
            int32_t soVal = (int32_t)stopOffset;
            int32_t oBBox = (int32_t)gd->objectBBox;
            memcpy(&gRow[0], &typeVal, 4);
            memcpy(&gRow[1], &spreadVal, 4);
            memcpy(&gRow[2], &scVal, 4);
            memcpy(&gRow[3], &soVal, 4);
            memcpy(&gRow[4], &oBBox, 4);
            gRow[5] = 0; gRow[6] = 0; gRow[7] = 0; /* pads */

            /* Next 16 words: float fields */
            float *gf = (float *)&gRow[8];
            gf[0]  = gd->x1;  gf[1]  = gd->y1;  gf[2]  = gd->x2;  gf[3]  = gd->y2;
            gf[4]  = gd->cx;  gf[5]  = gd->cy;  gf[6]  = gd->r;
            gf[7]  = gd->fx;  gf[8]  = gd->fy;  gf[9]  = gd->fr;
            gf[10] = gd->transform[0]; gf[11] = gd->transform[1];
            gf[12] = gd->transform[2]; gf[13] = gd->transform[3];
            gf[14] = gd->transform[4]; gf[15] = gd->transform[5];

            /* Fill stops */
            for (int si = 0; si < gd->stop_count; si++) {
                float *sRow = &stopBufData[(stopOffset + (uint32_t)si) * 8];
                sRow[0] = gd->stops[si].color[0];
                sRow[1] = gd->stops[si].color[1];
                sRow[2] = gd->stops[si].color[2];
                sRow[3] = gd->stops[si].color[3];
                sRow[4] = gd->stops[si].offset;
                sRow[5] = 0; sRow[6] = 0; sRow[7] = 0; /* pad */
            }
            stopOffset += (uint32_t)gd->stop_count;
        }
    }

    /* ---- Allocate SigilGPUScene ---- */
    SigilGPUScene *gs = (SigilGPUScene *)calloc(1, sizeof(SigilGPUScene));
    gs->elementCount = ec;
    gs->totalCurves = totalCurves;
    gs->gradientCount = (uint32_t)gradCount;
    gs->hasViewBox = scene->has_viewBox;
    memcpy(gs->viewBox, scene->viewBox, sizeof(float) * 4);
    gs->par_align = scene->par_align;
    gs->par_meet_or_slice = scene->par_meet_or_slice;
    gs->par_none = scene->par_none;

    /* Deep-copy gradient defs for CPU ramp baking */
    if (gradCount > 0) {
        gs->cpuGradients = (SigilGradientDef *)malloc((size_t)gradCount * sizeof(SigilGradientDef));
        memcpy(gs->cpuGradients, scene->gradients, (size_t)gradCount * sizeof(SigilGradientDef));
        for (int gi = 0; gi < gradCount; gi++) {
            if (gs->cpuGradients[gi].stop_count > 0) {
                size_t sz = (size_t)gs->cpuGradients[gi].stop_count * sizeof(SigilGradientStop);
                gs->cpuGradients[gi].stops = (SigilGradientStop *)malloc(sz);
                memcpy(gs->cpuGradients[gi].stops, scene->gradients[gi].stops, sz);
            }
        }
    }

    WGPUDevice device = ctx->device;
    WGPUQueue queue = ctx->queue;

    /* ---- Create and upload input buffers ---- */

    /* Curves buffer */
    WGPUBufferDescriptor curvesBufDesc = {
        .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        .size = curvesBytes, .nextInChain = NULL, .label = {NULL, WGPU_STRLEN}};
    gs->curvesBuf = wgpuDeviceCreateBuffer(device, &curvesBufDesc);
    wgpuQueueWriteBuffer(queue, gs->curvesBuf, 0, curvesData, curvesBytes);

    /* Elements buffer */
    WGPUBufferDescriptor elementsBufDesc = {
        .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        .size = elemBytes, .nextInChain = NULL, .label = {NULL, WGPU_STRLEN}};
    gs->elementsBuf = wgpuDeviceCreateBuffer(device, &elementsBufDesc);
    wgpuQueueWriteBuffer(queue, gs->elementsBuf, 0, elemData, elemBytes);

    /* Offsets buffer */
    WGPUBufferDescriptor offsetsBufDesc = {
        .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        .size = offsetsBytes, .nextInChain = NULL, .label = {NULL, WGPU_STRLEN}};
    gs->offsetsBuf = wgpuDeviceCreateBuffer(device, &offsetsBufDesc);
    wgpuQueueWriteBuffer(queue, gs->offsetsBuf, 0, offsetData, offsetsBytes);

    /* Gradients buffer */
    WGPUBufferDescriptor gradientsBufDesc = {
        .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        .size = gradBufBytes, .nextInChain = NULL, .label = {NULL, WGPU_STRLEN}};
    gs->gradientsBuf = wgpuDeviceCreateBuffer(device, &gradientsBufDesc);
    if (gradBufData) {
        wgpuQueueWriteBuffer(queue, gs->gradientsBuf, 0, gradBufData, gradBufBytes);
    }

    /* Gradient stops buffer */
    WGPUBufferDescriptor gradStopsBufDesc = {
        .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        .size = stopBufBytes, .nextInChain = NULL, .label = {NULL, WGPU_STRLEN}};
    gs->gradientStopsBuf = wgpuDeviceCreateBuffer(device, &gradStopsBufDesc);
    if (stopBufData) {
        wgpuQueueWriteBuffer(queue, gs->gradientStopsBuf, 0, stopBufData, stopBufBytes);
    }

    /* Viewport uniform buffer (32 bytes = 8 floats, written per-prepare) */
    WGPUBufferDescriptor viewportBufDesc = {
        .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
        .size = 32, .nextInChain = NULL, .label = {NULL, WGPU_STRLEN}};
    gs->viewportBuf = wgpuDeviceCreateBuffer(device, &viewportBufDesc);

    /* ---- Create output buffers ---- */
    /* All output buffers get CopyDst for CPU fallback path */

    /* curveBuf: curveOutVec4s * 16 bytes */
    uint64_t curveBufSize = curveOutVec4s > 0 ? (uint64_t)curveOutVec4s * 16 : 4;
    WGPUBufferDescriptor curveBufDesc = {
        .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        .size = curveBufSize, .nextInChain = NULL, .label = {NULL, WGPU_STRLEN}};
    gs->curveBuf = wgpuDeviceCreateBuffer(device, &curveBufDesc);

    /* bandBuf: bandOutVec4s * 16 bytes */
    uint64_t bandBufSize = bandOutVec4s > 0 ? (uint64_t)bandOutVec4s * 16 : 4;
    WGPUBufferDescriptor bandBufDesc = {
        .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        .size = bandBufSize, .nextInChain = NULL, .label = {NULL, WGPU_STRLEN}};
    gs->bandBuf = wgpuDeviceCreateBuffer(device, &bandBufDesc);

    /* vertexBuf: ec * 4 * 112 bytes (Vertex + CopyDst) */
    uint64_t vertexBufSize = (uint64_t)ec * 4 * 112;
    if (vertexBufSize < 4) vertexBufSize = 4;
    WGPUBufferDescriptor vertexBufDesc = {
        .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc,
        .size = vertexBufSize, .nextInChain = NULL, .label = {NULL, WGPU_STRLEN}};
    gs->vertexBuf = wgpuDeviceCreateBuffer(device, &vertexBufDesc);

    /* indexBuf: ec * 6 * 4 bytes (Index + CopyDst) */
    uint64_t indexBufSize = (uint64_t)ec * 6 * 4;
    if (indexBufSize < 4) indexBufSize = 4;
    WGPUBufferDescriptor indexBufDesc = {
        .usage = WGPUBufferUsage_Index | WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        .size = indexBufSize, .nextInChain = NULL, .label = {NULL, WGPU_STRLEN}};
    gs->indexBuf = wgpuDeviceCreateBuffer(device, &indexBufDesc);

    /* gradientRampBuf: gradTexH * 256 * 4 bytes (Storage + CopySrc) */
    uint64_t rampBufSize = (uint64_t)gradTexH * 256 * 4;
    if (rampBufSize < 4) rampBufSize = 4;
    WGPUBufferDescriptor gradRampBufDesc = {
        .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc,
        .size = rampBufSize, .nextInChain = NULL, .label = {NULL, WGPU_STRLEN}};
    gs->gradientRampBuf = wgpuDeviceCreateBuffer(device, &gradRampBufDesc);

    /* ---- Gradient ramp texture: RGBA8Unorm, 256 x gradTexH ---- */
    WGPUTextureDescriptor gradTexDesc = {
        .usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
        .size = {(uint32_t)SIGIL_GRADIENT_RAMP_WIDTH, (uint32_t)gradTexH, 1},
        .format = WGPUTextureFormat_RGBA8Unorm,
        .mipLevelCount = 1, .sampleCount = 1,
        .dimension = WGPUTextureDimension_2D,
        .nextInChain = NULL, .label = {NULL, WGPU_STRLEN}};
    gs->gradientTexture = wgpuDeviceCreateTexture(device, &gradTexDesc);
    WGPUTextureViewDescriptor gradViewDesc = {
        .format = WGPUTextureFormat_RGBA8Unorm,
        .dimension = WGPUTextureViewDimension_2D,
        .mipLevelCount = 1, .arrayLayerCount = 1,
        .aspect = WGPUTextureAspect_All,
        .usage = WGPUTextureUsage_TextureBinding,
        .nextInChain = NULL, .label = {NULL, WGPU_STRLEN}};
    gs->gradientView = wgpuTextureCreateView(gs->gradientTexture, &gradViewDesc);

    /* ---- Build compute bind groups ---- */

    /* prepareInputBG: group(0) — 5 bindings */
    WGPUBindGroupEntry inputEntries[5] = {
        {.binding = 0, .buffer = gs->curvesBuf,    .size = curvesBytes, .nextInChain = NULL, .offset = 0},
        {.binding = 1, .buffer = gs->elementsBuf,  .size = elemBytes, .nextInChain = NULL, .offset = 0},
        {.binding = 2, .buffer = gs->offsetsBuf,   .size = offsetsBytes, .nextInChain = NULL, .offset = 0},
        {.binding = 3, .buffer = gs->gradientsBuf, .size = gradBufBytes, .nextInChain = NULL, .offset = 0},
        {.binding = 4, .buffer = gs->viewportBuf,  .size = 32, .nextInChain = NULL, .offset = 0},
    };
    WGPUBindGroupDescriptor prepInBGDesc = {
        .layout = ctx->prepareInputBGL,
        .entryCount = 5, .entries = inputEntries,
        .nextInChain = NULL, .label = {NULL, WGPU_STRLEN}};
    gs->prepareInputBG = wgpuDeviceCreateBindGroup(device, &prepInBGDesc);

    /* prepareOutputBG: group(1) — 4 bindings */
    WGPUBindGroupEntry outputEntries[4] = {
        {.binding = 0, .buffer = gs->curveBuf,  .size = curveBufSize, .nextInChain = NULL, .offset = 0},
        {.binding = 1, .buffer = gs->bandBuf,   .size = bandBufSize, .nextInChain = NULL, .offset = 0},
        {.binding = 2, .buffer = gs->vertexBuf, .size = vertexBufSize, .nextInChain = NULL, .offset = 0},
        {.binding = 3, .buffer = gs->indexBuf,  .size = indexBufSize, .nextInChain = NULL, .offset = 0},
    };
    WGPUBindGroupDescriptor prepOutBGDesc = {
        .layout = ctx->prepareOutputBGL,
        .entryCount = 4, .entries = outputEntries,
        .nextInChain = NULL, .label = {NULL, WGPU_STRLEN}};
    gs->prepareOutputBG = wgpuDeviceCreateBindGroup(device, &prepOutBGDesc);

    /* gradientBG: group(0) of gradient compute — 3 bindings (NULL if no grads) */
    if (gradCount > 0) {
        WGPUBindGroupEntry gradEntries[3] = {
            {.binding = 0, .buffer = gs->gradientsBuf,    .size = gradBufBytes, .nextInChain = NULL, .offset = 0},
            {.binding = 1, .buffer = gs->gradientStopsBuf, .size = stopBufBytes, .nextInChain = NULL, .offset = 0},
            {.binding = 2, .buffer = gs->gradientRampBuf,  .size = rampBufSize, .nextInChain = NULL, .offset = 0},
        };
        WGPUBindGroupDescriptor gradBGDesc = {
            .layout = ctx->gradientBGL,
            .entryCount = 3, .entries = gradEntries,
            .nextInChain = NULL, .label = {NULL, WGPU_STRLEN}};
        gs->gradientBG = wgpuDeviceCreateBindGroup(device, &gradBGDesc);
    }

    /* ---- Store CPU copies for CPU prepare fallback ---- */
    gs->cpuElemData   = elemData;     /* 22 u32 per element */
    gs->cpuOffsetData = offsetData;   /* 2 u32 per element */
    gs->cpuCurvesData = curvesData;   /* 6 floats per curve */
    gs->curveOutVec4s = curveOutVec4s;
    gs->bandOutVec4s  = bandOutVec4s;

    /* ---- Free gradient staging only (elem/curve/offset kept) ---- */
    free(gradBufData);
    free(stopBufData);

    return gs;
}

/* ------------------------------------------------------------------ */
/*  sigil_free_gpu_scene — release all GPU resources                  */
/* ------------------------------------------------------------------ */

void sigil_free_gpu_scene(SigilGPUScene* gs) {
    if (!gs) return;

    /* Bind groups */
    if (gs->prepareInputBG)  wgpuBindGroupRelease(gs->prepareInputBG);
    if (gs->prepareOutputBG) wgpuBindGroupRelease(gs->prepareOutputBG);
    if (gs->gradientBG)      wgpuBindGroupRelease(gs->gradientBG);

    /* Texture view + texture */
    if (gs->gradientView)    wgpuTextureViewRelease(gs->gradientView);
    if (gs->gradientTexture) wgpuTextureDestroy(gs->gradientTexture);
    if (gs->gradientTexture) wgpuTextureRelease(gs->gradientTexture);

    /* Input buffers */
    if (gs->curvesBuf)       { wgpuBufferDestroy(gs->curvesBuf);       wgpuBufferRelease(gs->curvesBuf); }
    if (gs->elementsBuf)     { wgpuBufferDestroy(gs->elementsBuf);     wgpuBufferRelease(gs->elementsBuf); }
    if (gs->offsetsBuf)      { wgpuBufferDestroy(gs->offsetsBuf);      wgpuBufferRelease(gs->offsetsBuf); }
    if (gs->gradientsBuf)    { wgpuBufferDestroy(gs->gradientsBuf);    wgpuBufferRelease(gs->gradientsBuf); }
    if (gs->gradientStopsBuf){ wgpuBufferDestroy(gs->gradientStopsBuf);wgpuBufferRelease(gs->gradientStopsBuf); }
    if (gs->viewportBuf)     { wgpuBufferDestroy(gs->viewportBuf);     wgpuBufferRelease(gs->viewportBuf); }

    /* Output buffers */
    if (gs->curveBuf)        { wgpuBufferDestroy(gs->curveBuf);        wgpuBufferRelease(gs->curveBuf); }
    if (gs->bandBuf)         { wgpuBufferDestroy(gs->bandBuf);         wgpuBufferRelease(gs->bandBuf); }
    if (gs->vertexBuf)       { wgpuBufferDestroy(gs->vertexBuf);       wgpuBufferRelease(gs->vertexBuf); }
    if (gs->indexBuf)        { wgpuBufferDestroy(gs->indexBuf);        wgpuBufferRelease(gs->indexBuf); }
    if (gs->gradientRampBuf) { wgpuBufferDestroy(gs->gradientRampBuf); wgpuBufferRelease(gs->gradientRampBuf); }

    /* CPU-side copies */
    if (gs->cpuGradients) {
        for (uint32_t gi = 0; gi < gs->gradientCount; gi++)
            free(gs->cpuGradients[gi].stops);
        free(gs->cpuGradients);
    }
    free(gs->cpuElemData);
    free(gs->cpuOffsetData);
    free(gs->cpuCurvesData);

    free(gs);
}

/* ------------------------------------------------------------------ */
/*  sigil_prepare_gpu — compute dispatch + draw-data creation          */
/* ------------------------------------------------------------------ */

SigilDrawData* sigil_prepare_gpu(SigilContext* ctx, SigilGPUScene* gs,
                                 WGPUCommandEncoder encoder,
                                 float viewport_w, float viewport_h) {
    if (!ctx || !gs || !encoder) return NULL;
    WGPUQueue queue = ctx->queue;

    /* ---- 1. Viewport computation ---- */
    float vbX, vbY, vbW, vbH;
    if (gs->hasViewBox) {
        vbX = gs->viewBox[0];
        vbY = gs->viewBox[1];
        vbW = gs->viewBox[2];
        vbH = gs->viewBox[3];
    } else {
        vbX = 0.0f;
        vbY = 0.0f;
        vbW = viewport_w;
        vbH = viewport_h;
    }
    float scaleX, scaleY, parOX = 0, parOY = 0;
    if (vbW > 0 && vbH > 0) {
        float sx = viewport_w / vbW, sy = viewport_h / vbH;
        if (gs->par_none) {
            scaleX = sx; scaleY = sy;
        } else {
            float s = gs->par_meet_or_slice ? fmaxf(sx, sy) : fminf(sx, sy);
            scaleX = s; scaleY = s;
            float dx = viewport_w - vbW * s;
            float dy = viewport_h - vbH * s;
            int xa = gs->par_align % 3, ya = gs->par_align / 3;
            parOX = xa == 0 ? 0 : (xa == 1 ? dx * 0.5f : dx);
            parOY = ya == 0 ? 0 : (ya == 1 ? dy * 0.5f : dy);
        }
    } else {
        scaleX = 1.0f; scaleY = 1.0f;
    }
    float invScaleX = (scaleX > 0.0f) ? 1.0f / scaleX : 1.0f;
    float invScaleY = (scaleY > 0.0f) ? 1.0f / scaleY : 1.0f;

    uint32_t ec = gs->elementCount;
    uint32_t *elemData   = gs->cpuElemData;
    uint32_t *offsetData = gs->cpuOffsetData;
    float    *curvesData = gs->cpuCurvesData;

    /* ---- 2. CPU prepare: generate curveBuf, bandBuf, vertexBuf, indexBuf ---- */

    /* 2a. Pack curves into curveBuf (2 vec4s per curve = 8 floats) */
    uint32_t curveOutVec4s = gs->curveOutVec4s;
    size_t curveBufFloats = (size_t)curveOutVec4s * 4;
    float *cpuCurveBuf = (float *)calloc(curveBufFloats > 0 ? curveBufFloats : 1, sizeof(float));

    for (uint32_t ei = 0; ei < ec; ei++) {
        uint32_t *row = &elemData[ei * 22];
        uint32_t curveOff   = row[0]; /* curve_offset into curvesData */
        uint32_t nc         = row[1]; /* curve_count */
        uint32_t curveStart = offsetData[ei * 2 + 0]; /* output offset in vec4s */

        for (uint32_t ci = 0; ci < nc; ci++) {
            float *c = &curvesData[(curveOff + ci) * 6];
            uint32_t base = (curveStart + ci * 2) * 4;
            cpuCurveBuf[base + 0] = c[0]; /* p0x */
            cpuCurveBuf[base + 1] = c[1]; /* p0y */
            cpuCurveBuf[base + 2] = c[2]; /* p1x */
            cpuCurveBuf[base + 3] = c[3]; /* p1y */
            cpuCurveBuf[base + 4] = c[4]; /* p2x */
            cpuCurveBuf[base + 5] = c[5]; /* p2y */
            cpuCurveBuf[base + 6] = 0.0f;
            cpuCurveBuf[base + 7] = 0.0f;
        }
    }

    /* 2b. Build bands in bandBuf */
    #define SIGIL_BAND_CNT 8
    uint32_t bandOutVec4s = gs->bandOutVec4s;
    size_t bandBufU32s = (size_t)bandOutVec4s * 4;
    uint32_t *cpuBandBuf = (uint32_t *)calloc(bandBufU32s > 0 ? bandBufU32s : 1, sizeof(uint32_t));

    for (uint32_t ei = 0; ei < ec; ei++) {
        uint32_t *row = &elemData[ei * 22];
        uint32_t nc = row[1];
        uint32_t curveOff = row[0];

        float xMin, yMin, xMax, yMax;
        memcpy(&xMin, &row[2], 4); memcpy(&yMin, &row[3], 4);
        memcpy(&xMax, &row[4], 4); memcpy(&yMax, &row[5], 4);
        float ew = xMax - xMin, eh = yMax - yMin;
        float bsX = ew > 0 ? 8.0f / ew : 0.0f;
        float bsY = eh > 0 ? 8.0f / eh : 0.0f;
        float boX = -xMin * bsX, boY = -yMin * bsY;

        uint32_t bandStart = offsetData[ei * 2 + 1];
        uint32_t curveStart = offsetData[ei * 2 + 0];
        uint32_t bandDataBase = bandStart + 16; /* past 16 headers */

        /* Band counters */
        uint32_t hCnt[SIGIL_BAND_CNT] = {0};
        uint32_t vCnt[SIGIL_BAND_CNT] = {0};

        for (uint32_t ci = 0; ci < nc; ci++) {
            float *c = &curvesData[(curveOff + ci) * 6];
            float cxMin = fminf(fminf(c[0], c[2]), c[4]);
            float cxMax = fmaxf(fmaxf(c[0], c[2]), c[4]);
            float cyMin = fminf(fminf(c[1], c[3]), c[5]);
            float cyMax = fmaxf(fmaxf(c[1], c[3]), c[5]);

            uint32_t curveRef = curveStart + ci * 2;

            /* Horizontal bands (partition by Y) */
            if (eh > 0) {
                int b0 = (int)fmaxf(0, fminf(7, floorf(cyMin * bsY + boY)));
                int b1 = (int)fmaxf(0, fminf(7, floorf(cyMax * bsY + boY)));
                for (int b = b0; b <= b1; b++) {
                    uint32_t slot = (bandDataBase + (uint32_t)b * nc + hCnt[b]) * 4;
                    cpuBandBuf[slot] = curveRef;
                    hCnt[b]++;
                }
            }

            /* Vertical bands (partition by X) */
            if (ew > 0) {
                int b0 = (int)fmaxf(0, fminf(7, floorf(cxMin * bsX + boX)));
                int b1 = (int)fmaxf(0, fminf(7, floorf(cxMax * bsX + boX)));
                for (int b = b0; b <= b1; b++) {
                    uint32_t slot = (bandDataBase + (SIGIL_BAND_CNT + (uint32_t)b) * nc + vCnt[b]) * 4;
                    cpuBandBuf[slot] = curveRef;
                    vCnt[b]++;
                }
            }
        }

        /* Sort bands: insertion sort descending by max coordinate */
        for (int b = 0; b < SIGIL_BAND_CNT; b++) {
            uint32_t cnt = hCnt[b];
            uint32_t base = (bandDataBase + (uint32_t)b * nc) * 4;
            for (uint32_t i = 1; i < cnt; i++) {
                uint32_t keyRef = cpuBandBuf[base + i * 4];
                uint32_t keySlot[4] = {cpuBandBuf[base+i*4], cpuBandBuf[base+i*4+1],
                                       cpuBandBuf[base+i*4+2], cpuBandBuf[base+i*4+3]};
                float keyVal = fmaxf(fmaxf(cpuCurveBuf[keyRef*4], cpuCurveBuf[keyRef*4+2]),
                                     cpuCurveBuf[(keyRef+1)*4]);
                uint32_t j = i;
                while (j > 0) {
                    uint32_t prevRef = cpuBandBuf[base + (j-1)*4];
                    float prevVal = fmaxf(fmaxf(cpuCurveBuf[prevRef*4], cpuCurveBuf[prevRef*4+2]),
                                          cpuCurveBuf[(prevRef+1)*4]);
                    if (prevVal >= keyVal) break;
                    memcpy(&cpuBandBuf[base+j*4], &cpuBandBuf[base+(j-1)*4], 16);
                    j--;
                }
                memcpy(&cpuBandBuf[base+j*4], keySlot, 16);
            }
        }
        for (int b = 0; b < SIGIL_BAND_CNT; b++) {
            uint32_t cnt = vCnt[b];
            uint32_t base = (bandDataBase + (SIGIL_BAND_CNT + (uint32_t)b) * nc) * 4;
            for (uint32_t i = 1; i < cnt; i++) {
                uint32_t keyRef = cpuBandBuf[base + i * 4];
                uint32_t keySlot[4] = {cpuBandBuf[base+i*4], cpuBandBuf[base+i*4+1],
                                       cpuBandBuf[base+i*4+2], cpuBandBuf[base+i*4+3]};
                float keyVal = fmaxf(fmaxf(cpuCurveBuf[keyRef*4+1], cpuCurveBuf[keyRef*4+3]),
                                     cpuCurveBuf[(keyRef+1)*4+1]);
                uint32_t j = i;
                while (j > 0) {
                    uint32_t prevRef = cpuBandBuf[base + (j-1)*4];
                    float prevVal = fmaxf(fmaxf(cpuCurveBuf[prevRef*4+1], cpuCurveBuf[prevRef*4+3]),
                                          cpuCurveBuf[(prevRef+1)*4+1]);
                    if (prevVal >= keyVal) break;
                    memcpy(&cpuBandBuf[base+j*4], &cpuBandBuf[base+(j-1)*4], 16);
                    j--;
                }
                memcpy(&cpuBandBuf[base+j*4], keySlot, 16);
            }
        }

        /* Write band headers */
        for (int b = 0; b < SIGIL_BAND_CNT; b++) {
            uint32_t idx = (bandStart + (uint32_t)b) * 4;
            cpuBandBuf[idx + 0] = hCnt[b];
            cpuBandBuf[idx + 1] = 16 + (uint32_t)b * nc;
            cpuBandBuf[idx + 2] = 0;
            cpuBandBuf[idx + 3] = 0;
        }
        for (int b = 0; b < SIGIL_BAND_CNT; b++) {
            uint32_t idx = (bandStart + SIGIL_BAND_CNT + (uint32_t)b) * 4;
            cpuBandBuf[idx + 0] = vCnt[b];
            cpuBandBuf[idx + 1] = 16 + (SIGIL_BAND_CNT + (uint32_t)b) * nc;
            cpuBandBuf[idx + 2] = 0;
            cpuBandBuf[idx + 3] = 0;
        }
    }
    #undef SIGIL_BAND_CNT

    /* 2c. Generate vertices + indices */
    uint32_t vertFloats = ec * 4 * 28;
    float *cpuVertBuf = (float *)calloc(vertFloats > 0 ? vertFloats : 1, sizeof(float));
    uint32_t *cpuIdxBuf = (uint32_t *)calloc(ec * 6 > 0 ? ec * 6 : 1, sizeof(uint32_t));

    for (uint32_t ei = 0; ei < ec; ei++) {
        uint32_t *row = &elemData[ei * 22];
        uint32_t nc = row[1];
        float xMin, yMin, xMax, yMax;
        memcpy(&xMin, &row[2], 4); memcpy(&yMin, &row[3], 4);
        memcpy(&xMax, &row[4], 4); memcpy(&yMax, &row[5], 4);
        float ew = xMax - xMin, eh = yMax - yMin;
        float bsX = ew > 0 ? 8.0f / ew : 0.0f;
        float bsY = eh > 0 ? 8.0f / eh : 0.0f;
        float boX = -xMin * bsX, boY = -yMin * bsY;

        float px0 = (xMin - vbX) * scaleX + parOX;
        float py0 = (yMin - vbY) * scaleY + parOY;
        float px1 = (xMax - vbX) * scaleX + parOX;
        float py1 = (yMax - vbY) * scaleY + parOY;

        uint32_t bandStart = offsetData[ei * 2 + 1];
        uint32_t glpU32 = bandStart;
        float glp; memcpy(&glp, &glpU32, 4);

        uint32_t bandMaxPacked = (7u << 16u) | 7u;
        if (row[10] == 1) bandMaxPacked |= (1u << 28u); /* evenodd */
        float bmp; memcpy(&bmp, &bandMaxPacked, 4);

        float fill_r, fill_g, fill_b, fill_a, opacity;
        memcpy(&fill_r, &row[6], 4); memcpy(&fill_g, &row[7], 4);
        memcpy(&fill_b, &row[8], 4); memcpy(&fill_a, &row[9], 4);
        memcpy(&opacity, &row[12], 4);

        float alpha = fill_a * opacity;
        float cr = fill_r * alpha, cg = fill_g * alpha, cb = fill_b * alpha;

        float g0[4], g1[4];
        memcpy(g0, &row[14], 16);
        memcpy(g1, &row[18], 16);

        int32_t gradIdx; memcpy(&gradIdx, &row[11], 4);
        if (gradIdx >= 0) {
            cr = 0; cg = 0; cb = 0;
            alpha = -opacity;
        }

        /* 4 corners: (px, py, nx, ny, emx, emy, glp, bmp, jac[4], bnd[4], col[4], grad0[4], grad1[4]) */
        float corners[4][4] = {{px0,py0,-1,-1},{px1,py0,1,-1},{px1,py1,1,1},{px0,py1,-1,1}};
        float emCoords[4][2] = {{xMin,yMin},{xMax,yMin},{xMax,yMax},{xMin,yMax}};

        for (int v = 0; v < 4; v++) {
            float *vp = &cpuVertBuf[(ei * 4 + (uint32_t)v) * 28];
            vp[0] = corners[v][0]; vp[1] = corners[v][1];
            vp[2] = corners[v][2]; vp[3] = corners[v][3];
            vp[4] = emCoords[v][0]; vp[5] = emCoords[v][1];
            vp[6] = glp; vp[7] = bmp;
            vp[8] = invScaleX; vp[9] = 0; vp[10] = 0; vp[11] = invScaleY;
            vp[12] = bsX; vp[13] = bsY; vp[14] = boX; vp[15] = boY;
            vp[16] = cr; vp[17] = cg; vp[18] = cb; vp[19] = alpha;
            vp[20] = g0[0]; vp[21] = g0[1]; vp[22] = g0[2]; vp[23] = g0[3];
            vp[24] = g1[0]; vp[25] = g1[1]; vp[26] = g1[2]; vp[27] = g1[3];
        }

        uint32_t idxBase = ei * 6, vtxBase = ei * 4;
        cpuIdxBuf[idxBase + 0] = vtxBase;
        cpuIdxBuf[idxBase + 1] = vtxBase + 1;
        cpuIdxBuf[idxBase + 2] = vtxBase + 2;
        cpuIdxBuf[idxBase + 3] = vtxBase;
        cpuIdxBuf[idxBase + 4] = vtxBase + 2;
        cpuIdxBuf[idxBase + 5] = vtxBase + 3;
    }

    /* ---- 3. Upload CPU-generated buffers to GPU ---- */
    if (curveOutVec4s > 0)
        wgpuQueueWriteBuffer(queue, gs->curveBuf, 0, cpuCurveBuf, (size_t)curveOutVec4s * 16);
    if (bandOutVec4s > 0)
        wgpuQueueWriteBuffer(queue, gs->bandBuf, 0, cpuBandBuf, (size_t)bandOutVec4s * 16);
    if (vertFloats > 0)
        wgpuQueueWriteBuffer(queue, gs->vertexBuf, 0, cpuVertBuf, (size_t)vertFloats * sizeof(float));
    if (ec > 0)
        wgpuQueueWriteBuffer(queue, gs->indexBuf, 0, cpuIdxBuf, (size_t)ec * 6 * sizeof(uint32_t));

    free(cpuCurveBuf);
    free(cpuBandBuf);
    free(cpuVertBuf);
    free(cpuIdxBuf);

    /* ---- 3b. CPU gradient ramp baking ---- */
    if (gs->gradientCount > 0 && gs->cpuGradients) {
        uint32_t gradTexH = gs->gradientCount;
        uint8_t *rampPixels = (uint8_t *)calloc((size_t)gradTexH * 256, 4);

        for (uint32_t gi = 0; gi < gradTexH; gi++) {
            const SigilGradientDef *gd = &gs->cpuGradients[gi];
            int sc = gd->stop_count;

            for (int x = 0; x < 256; x++) {
                float r = 0, g = 0, b = 0, a = 0;
                float t = (float)x / 255.0f;

                if (sc == 1) {
                    r = gd->stops[0].color[0]; g = gd->stops[0].color[1];
                    b = gd->stops[0].color[2]; a = gd->stops[0].color[3];
                } else if (sc >= 2) {
                    /* Clamp to edges */
                    if (t <= gd->stops[0].offset) {
                        r = gd->stops[0].color[0]; g = gd->stops[0].color[1];
                        b = gd->stops[0].color[2]; a = gd->stops[0].color[3];
                    } else if (t >= gd->stops[sc-1].offset) {
                        r = gd->stops[sc-1].color[0]; g = gd->stops[sc-1].color[1];
                        b = gd->stops[sc-1].color[2]; a = gd->stops[sc-1].color[3];
                    } else {
                        /* Find bracketing stops */
                        int lo = 0;
                        for (int s = 0; s < sc - 1; s++) {
                            if (t >= gd->stops[s].offset && t <= gd->stops[s+1].offset) {
                                lo = s; break;
                            }
                        }
                        float seg = gd->stops[lo+1].offset - gd->stops[lo].offset;
                        float u = (seg > 1e-6f) ? (t - gd->stops[lo].offset) / seg : 0.0f;
                        if (u < 0) u = 0; if (u > 1) u = 1;
                        r = gd->stops[lo].color[0] * (1-u) + gd->stops[lo+1].color[0] * u;
                        g = gd->stops[lo].color[1] * (1-u) + gd->stops[lo+1].color[1] * u;
                        b = gd->stops[lo].color[2] * (1-u) + gd->stops[lo+1].color[2] * u;
                        a = gd->stops[lo].color[3] * (1-u) + gd->stops[lo+1].color[3] * u;
                    }
                }

                uint8_t *px = &rampPixels[(gi * 256 + (uint32_t)x) * 4];
                px[0] = (uint8_t)(r < 0 ? 0 : (r > 1 ? 255 : (int)(r * 255.0f + 0.5f)));
                px[1] = (uint8_t)(g < 0 ? 0 : (g > 1 ? 255 : (int)(g * 255.0f + 0.5f)));
                px[2] = (uint8_t)(b < 0 ? 0 : (b > 1 ? 255 : (int)(b * 255.0f + 0.5f)));
                px[3] = (uint8_t)(a < 0 ? 0 : (a > 1 ? 255 : (int)(a * 255.0f + 0.5f)));
            }
        }

        /* Upload ramp directly to the gradient texture */
        WGPUTexelCopyTextureInfo rampCopyTex = {
            .texture = gs->gradientTexture,
            .mipLevel = 0,
            .aspect = WGPUTextureAspect_All,
            .origin = {0, 0, 0}};
        WGPUTexelCopyBufferLayout rampCopyLayout = {
            .bytesPerRow = 256 * 4,
            .rowsPerImage = gradTexH,
            .offset = 0};
        WGPUExtent3D rampCopyExtent = {256, gradTexH, 1};
        wgpuQueueWriteTexture(queue,
            &rampCopyTex,
            rampPixels,
            (size_t)gradTexH * 256 * 4,
            &rampCopyLayout,
            &rampCopyExtent
        );
        free(rampPixels);
    }

    /* ---- 4. Create SigilDrawData ---- */
    SigilDrawData *dd = (SigilDrawData *)calloc(1, sizeof *dd);
    if (!dd) return NULL;

    dd->indexCount = gs->elementCount * 6;
    dd->vertexBuf  = gs->vertexBuf;   /* borrowed, not owned */
    dd->indexBuf   = gs->indexBuf;     /* borrowed, not owned */

    /* 80-byte uniform buffer (MVP + viewport size) */
    WGPUBufferDescriptor uboDesc = {
        .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
        .size  = 80, .nextInChain = NULL, .label = {NULL, WGPU_STRLEN}};
    dd->uniformBuffer = wgpuDeviceCreateBuffer(ctx->device, &uboDesc);

    float ubo[20] = {
         2.0f / viewport_w, 0, 0, -1.0f,
         0, -2.0f / viewport_h, 0,  1.0f,
         0, 0, 0, 0,
         0, 0, 0, 1,
         viewport_w, viewport_h, 0, 0,
    };
    wgpuQueueWriteBuffer(queue, dd->uniformBuffer, 0, ubo, sizeof ubo);

    /* Render bind group: uniform + curveBuf + bandBuf + gradientView + sampler */
    WGPUBindGroupEntry bgEntries[5] = {
        {.binding = 0, .buffer = dd->uniformBuffer,
         .size = 80, .nextInChain = NULL, .offset = 0},
        {.binding = 1, .buffer = gs->curveBuf,
         .size = wgpuBufferGetSize(gs->curveBuf), .nextInChain = NULL, .offset = 0},
        {.binding = 2, .buffer = gs->bandBuf,
         .size = wgpuBufferGetSize(gs->bandBuf), .nextInChain = NULL, .offset = 0},
        {.binding = 3, .textureView = gs->gradientView, .nextInChain = NULL},
        {.binding = 4, .sampler = ctx->gradientSampler, .nextInChain = NULL},
    };
    WGPUBindGroupDescriptor renderBGDesc = {
        .layout     = ctx->renderBGL,
        .entryCount = 5,
        .entries    = bgEntries,
        .nextInChain = NULL, .label = {NULL, WGPU_STRLEN}};
    dd->renderBindGroup = wgpuDeviceCreateBindGroup(ctx->device, &renderBGDesc);

    return dd;
}

/* sigil_prepare removed — replaced by sigil_upload + sigil_prepare_gpu */

/* ------------------------------------------------------------------ */
/*  sigil_encode — record render pass commands                        */
/* ------------------------------------------------------------------ */

void sigil_encode(SigilContext* ctx, SigilDrawData* data,
                  WGPUCommandEncoder encoder,
                  WGPUTextureView color_target,
                  WGPUTextureView depth_target,
                  const float clear_color[4]) {
    if (!ctx || !data || !encoder || !color_target) return;

    WGPUColor caClear;
    if (clear_color) {
        caClear.r = clear_color[0]; caClear.g = clear_color[1];
        caClear.b = clear_color[2]; caClear.a = clear_color[3];
    } else {
        caClear.r = 0; caClear.g = 0; caClear.b = 0; caClear.a = 1;
    }
    WGPURenderPassColorAttachment ca = {
        .view = color_target,
        .loadOp = clear_color ? WGPULoadOp_Clear : WGPULoadOp_Load,
        .storeOp = WGPUStoreOp_Store,
        .clearValue = caClear,
        .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
     .nextInChain = NULL};

    WGPURenderPassDepthStencilAttachment dsa;
    WGPURenderPassDescriptor rpDesc = {
        .colorAttachmentCount = 1,
        .colorAttachments = &ca,
     .nextInChain = NULL, .label = {NULL, WGPU_STRLEN}};

    if (depth_target) {
        memset(&dsa, 0, sizeof dsa);
        dsa.view = depth_target;
        dsa.depthLoadOp = WGPULoadOp_Clear;
        dsa.depthStoreOp = WGPUStoreOp_Store;
        dsa.depthClearValue = 1.0f;
        rpDesc.depthStencilAttachment = &dsa;
    }

    WGPURenderPassEncoder rp = wgpuCommandEncoderBeginRenderPass(encoder, &rpDesc);
    wgpuRenderPassEncoderSetPipeline(rp, ctx->pipeline);
    wgpuRenderPassEncoderSetBindGroup(rp, 0, data->renderBindGroup, 0, NULL);
    wgpuRenderPassEncoderSetVertexBuffer(rp, 0, data->vertexBuf, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(rp, data->indexBuf, WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDrawIndexed(rp, (uint32_t)data->indexCount, 1, 0, 0, 0);
    wgpuRenderPassEncoderEnd(rp);
    wgpuRenderPassEncoderRelease(rp);
}

/* ------------------------------------------------------------------ */
/*  sigil_free_draw_data — release GPU resources                      */
/* ------------------------------------------------------------------ */

void sigil_free_draw_data(SigilDrawData* data) {
    if (!data) return;
    if (data->renderBindGroup) wgpuBindGroupRelease(data->renderBindGroup);
    if (data->uniformBuffer) { wgpuBufferDestroy(data->uniformBuffer); wgpuBufferRelease(data->uniformBuffer); }
    /* vertexBuf and indexBuf are NOT released — owned by SigilGPUScene */
    free(data);
}

/* ------------------------------------------------------------------ */
/*  sigil_free_scene                                                  */
/* ------------------------------------------------------------------ */

void sigil_free_scene(SigilScene* scene) {
    if (!scene) return;
    for (int i = 0; i < scene->element_count; i++) {
        free(scene->elements[i].curves);
        for (int b = 0; b < SIGIL_BAND_COUNT; b++) {
            free(scene->elements[i].bands.hBands[b].curveIndices);
            free(scene->elements[i].bands.vBands[b].curveIndices);
        }
    }
    free(scene->elements);
    /* Free gradient data */
    for (int i = 0; i < scene->gradient_count; i++)
        free(scene->gradients[i].stops);
    free(scene->gradients);
    /* Free font data */
    for (int i = 0; i < scene->font_count; i++) {
        free(scene->font_names[i]);
    }
    free(scene->font_names);
    free(scene->fonts);
    free(scene);
}

/* ------------------------------------------------------------------ */
/*  sigil_destroy — release pipeline resources                        */
/* ------------------------------------------------------------------ */

void sigil_destroy(SigilContext* ctx) {
    if (!ctx) return;
    /* Render pipeline */
    if (ctx->pipeline)              wgpuRenderPipelineRelease(ctx->pipeline);
    if (ctx->renderPipelineLayout)  wgpuPipelineLayoutRelease(ctx->renderPipelineLayout);
    if (ctx->renderBGL)             wgpuBindGroupLayoutRelease(ctx->renderBGL);
    if (ctx->vertexShader)          wgpuShaderModuleRelease(ctx->vertexShader);
    if (ctx->fragmentShader)        wgpuShaderModuleRelease(ctx->fragmentShader);
    /* Compute pipelines */
    if (ctx->preparePipeline)       wgpuComputePipelineRelease(ctx->preparePipeline);
    if (ctx->gradientPipeline)      wgpuComputePipelineRelease(ctx->gradientPipeline);
    if (ctx->preparePipelineLayout) wgpuPipelineLayoutRelease(ctx->preparePipelineLayout);
    if (ctx->gradientPipelineLayout)wgpuPipelineLayoutRelease(ctx->gradientPipelineLayout);
    if (ctx->prepareInputBGL)       wgpuBindGroupLayoutRelease(ctx->prepareInputBGL);
    if (ctx->prepareOutputBGL)      wgpuBindGroupLayoutRelease(ctx->prepareOutputBGL);
    if (ctx->gradientBGL)           wgpuBindGroupLayoutRelease(ctx->gradientBGL);
    if (ctx->prepareShader)         wgpuShaderModuleRelease(ctx->prepareShader);
    if (ctx->gradientShader)        wgpuShaderModuleRelease(ctx->gradientShader);
    /* Shared */
    if (ctx->gradientSampler)       wgpuSamplerRelease(ctx->gradientSampler);
    free(ctx);
}

#ifdef __cplusplus
#  if defined(__clang__)
#    pragma clang diagnostic pop
#  elif defined(__GNUC__)
#    pragma GCC diagnostic pop
#  endif
} /* extern "C" */
#endif

#endif /* SIGIL_IMPLEMENTATION */
#endif /* SIGILVG_H */
