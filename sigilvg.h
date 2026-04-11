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

/* Resolve xlink:href inheritance between gradients */
static void sigil__resolve_gradient_hrefs(SigilGradientArray *arr) {
    for (int i = 0; i < arr->count; i++) {
        SigilGradientDef *g = &arr->data[i];
        if (g->href[0] == '\0') continue;
        /* Find referenced gradient */
        const char *ref = g->href;
        if (ref[0] == '#') ref++;
        for (int j = 0; j < arr->count; j++) {
            if (i == j) continue;
            if (strcmp(arr->data[j].id, ref) == 0) {
                /* Inherit stops if none defined */
                if (g->stop_count == 0 && arr->data[j].stop_count > 0) {
                    g->stop_count = arr->data[j].stop_count;
                    g->stops = (SigilGradientStop *)malloc((size_t)g->stop_count * sizeof(SigilGradientStop));
                    memcpy(g->stops, arr->data[j].stops, (size_t)g->stop_count * sizeof(SigilGradientStop));
                }
                break;
            }
        }
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

    /* CPU-side copies for CPU prepare path (compute shader fallback) */
    uint32_t *cpuElemData;         /* 22 u32/f32 per element */
    uint32_t *cpuOffsetData;       /* 2 u32 per element: curve_start, band_start */
    float    *cpuCurvesData;       /* 6 floats per curve */
    uint32_t  curveOutVec4s;       /* total vec4s in curveBuf */
    uint32_t  bandOutVec4s;        /* total vec4s in bandBuf */
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

/* Parse a CSS length value with optional unit suffix.
   ref_size is used for em/% (default font-size or viewport dimension).
   DPI assumption: 96 (CSS reference pixel). */
static float sigil__parse_length(const char *val, int vlen, float ref_size) {
    if (!val || vlen <= 0) return 0.0f;
    char *end;
    float v = strtof(val, &end);
    int remaining = vlen - (int)(end - val);
    while (remaining > 0 && isspace((unsigned char)*end)) { end++; remaining--; }
    if (remaining >= 2) {
        if (end[0]=='p' && end[1]=='x') return v;
        if (end[0]=='p' && end[1]=='t') return v * (96.0f / 72.0f);
        if (end[0]=='p' && end[1]=='c') return v * 16.0f;
        if (end[0]=='m' && end[1]=='m') return v * (96.0f / 25.4f);
        if (end[0]=='c' && end[1]=='m') return v * (96.0f / 2.54f);
        if (end[0]=='i' && end[1]=='n') return v * 96.0f;
        if (end[0]=='e' && end[1]=='m') return v * ref_size;
    }
    if (remaining >= 3 && end[0]=='r' && end[1]=='e' && end[2]=='m')
        return v * 16.0f; /* rem = root em, assume 16px */
    if (remaining >= 1 && end[0] == '%') return v * ref_size / 100.0f;
    return v; /* unitless = user units (px) */
}

static float sigil__get_attr_float(const char *attrs, int attrs_len,
                                    const char *name, float def) {
    const char *val;
    int vlen = sigil__get_attr(attrs, attrs_len, name, &val);
    if (vlen == 0 || !val) return def;
    return sigil__parse_length(val, vlen, 16.0f);
}

/* ------------------------------------------------------------------ */
/*  Color parsing                                                     */
/* ------------------------------------------------------------------ */

typedef struct { const char *name; float r, g, b; } SigilNamedColor;

static const SigilNamedColor sigil__named_colors[] = {
    {"black",   0.0f, 0.0f, 0.0f},
    {"white",   1.0f, 1.0f, 1.0f},
    {"red",     1.0f, 0.0f, 0.0f},
    {"green",   0.0f, 0.502f, 0.0f},
    {"blue",    0.0f, 0.0f, 1.0f},
    {"yellow",  1.0f, 1.0f, 0.0f},
    {"cyan",    0.0f, 1.0f, 1.0f},
    {"magenta", 1.0f, 0.0f, 1.0f},
    {"orange",  1.0f, 0.647f, 0.0f},
    {"purple",  0.502f, 0.0f, 0.502f},
    {"pink",    1.0f, 0.753f, 0.796f},
    {"brown",   0.647f, 0.165f, 0.165f},
    {"gray",    0.502f, 0.502f, 0.502f},
    {"grey",    0.502f, 0.502f, 0.502f},
    {"silver",  0.753f, 0.753f, 0.753f},
    {"gold",    1.0f, 0.843f, 0.0f},
    {"navy",    0.0f, 0.0f, 0.502f},
    {"teal",    0.0f, 0.502f, 0.502f},
    {"maroon",  0.502f, 0.0f, 0.0f},
    {"olive",   0.502f, 0.502f, 0.0f},
    {"lime",    0.0f, 1.0f, 0.0f},
    {"aqua",    0.0f, 1.0f, 1.0f},
    {"fuchsia", 1.0f, 0.0f, 1.0f},
    {"coral",   1.0f, 0.498f, 0.314f},
    {"salmon",  0.980f, 0.502f, 0.447f},
    {"tomato",  1.0f, 0.388f, 0.278f},
    {"crimson", 0.863f, 0.078f, 0.235f},
    {"indigo",  0.294f, 0.0f, 0.510f},
    {"violet",  0.933f, 0.510f, 0.933f},
    {NULL, 0, 0, 0}
};

/* Parse a hex digit (0-15). Returns -1 on failure. */
static int sigil__hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + c - 'a';
    if (c >= 'A' && c <= 'F') return 10 + c - 'A';
    return -1;
}

/* Parse CSS color string into float[4]. Returns 1 if valid color, 0 if "none". */
static int sigil__parse_color(const char *str, int len, float out[4]) {
    out[0] = out[1] = out[2] = 0.0f; out[3] = 1.0f;

    if (len == 0 || !str) return 0;

    /* Skip leading whitespace */
    while (len > 0 && isspace((unsigned char)*str)) { str++; len--; }
    while (len > 0 && isspace((unsigned char)str[len-1])) { len--; }

    if (len == 4 && memcmp(str, "none", 4) == 0) return 0;

    /* Hex color */
    if (str[0] == '#') {
        if (len == 4) { /* #RGB */
            int r = sigil__hex_digit(str[1]);
            int g = sigil__hex_digit(str[2]);
            int b = sigil__hex_digit(str[3]);
            if (r >= 0 && g >= 0 && b >= 0) {
                out[0] = (float)(r * 17) / 255.0f;
                out[1] = (float)(g * 17) / 255.0f;
                out[2] = (float)(b * 17) / 255.0f;
                return 1;
            }
        } else if (len == 7) { /* #RRGGBB */
            int r = sigil__hex_digit(str[1]) * 16 + sigil__hex_digit(str[2]);
            int g = sigil__hex_digit(str[3]) * 16 + sigil__hex_digit(str[4]);
            int b = sigil__hex_digit(str[5]) * 16 + sigil__hex_digit(str[6]);
            out[0] = (float)r / 255.0f;
            out[1] = (float)g / 255.0f;
            out[2] = (float)b / 255.0f;
            return 1;
        }
    }

    /* rgb(r,g,b) */
    if (len > 4 && memcmp(str, "rgb(", 4) == 0) {
        float r, g, b;
        if (sscanf(str + 4, "%f,%f,%f", &r, &g, &b) == 3 ||
            sscanf(str + 4, "%f %f %f", &r, &g, &b) == 3) {
            out[0] = r / 255.0f;
            out[1] = g / 255.0f;
            out[2] = b / 255.0f;
            return 1;
        }
    }

    /* Named colors */
    for (int i = 0; sigil__named_colors[i].name; i++) {
        int nlen = (int)strlen(sigil__named_colors[i].name);
        if (nlen == len && memcmp(str, sigil__named_colors[i].name, (size_t)len) == 0) {
            out[0] = sigil__named_colors[i].r;
            out[1] = sigil__named_colors[i].g;
            out[2] = sigil__named_colors[i].b;
            return 1;
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

static int sigil__rect_to_curves(float x, float y, float w, float h,
                                  float rx, float ry,
                                  SigilCurve **out, SigilBounds *bounds) {
    (void)rx; (void)ry; /* TODO: rounded corners */
    SigilCurveArray arr;
    sigil__curve_array_init(&arr);

    /* 4 sides: top, right, bottom, left */
    sigil__curve_array_push(&arr, sigil__line_to_quad(x, y, x + w, y));
    sigil__curve_array_push(&arr, sigil__line_to_quad(x + w, y, x + w, y + h));
    sigil__curve_array_push(&arr, sigil__line_to_quad(x + w, y + h, x, y + h));
    sigil__curve_array_push(&arr, sigil__line_to_quad(x, y + h, x, y));

    bounds->xMin = x; bounds->yMin = y;
    bounds->xMax = x + w; bounds->yMax = y + h;

    *out = arr.data;
    return arr.count;
}

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
       sin^2(theta/2) = (1 - dot_n)/2 */
    float sin2_half = (1.0f - dot_n) * 0.5f;
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

/* ---- Inner join: route through vertex to avoid self-intersection ---- */
static void sigil__inner_join(SigilCurveArray *arr,
                               float vx, float vy,
                               float ipx, float ipy,
                               float inx, float iny) {
    sigil__curve_array_push(arr, sigil__line_to_quad(ipx, ipy, vx, vy));
    sigil__curve_array_push(arr, sigil__line_to_quad(vx, vy, inx, iny));
}

/* ================================================================== */

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
                    /* Right turn: right side is inner — route through vertex */
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

    int pos = 0;
    SigilTag tag;

    while (sigil__next_tag(svg_data, (int)len, &pos, &tag)) {
        if (tag.is_close) {
            /* Closing tag */
            if (sigil__tag_is(&tag, "g")) {
                if (xform_depth > 0) xform_depth--;
            }
            else if (sigil__tag_is(&tag, "defs")) { in_defs = 0; }
            else if (sigil__tag_is(&tag, "linearGradient") ||
                     sigil__tag_is(&tag, "radialGradient")) { in_gradient = 0; }
            continue;
        }

        /* <defs> section */
        if (sigil__tag_is(&tag, "defs")) {
            in_defs = 1;
            continue;
        }

        /* <linearGradient> and <radialGradient> (can appear inside or outside <defs>) */
        if (sigil__tag_is(&tag, "linearGradient")) {
            SigilGradientDef *g = sigil__grad_array_push(&grad_defs);
            g->type = 1;
            /* Extract id */
            const char *id_val;
            int id_len = sigil__get_attr(tag.attrs, tag.attrs_len, "id", &id_val);
            if (id_len > 0 && id_len < 127) { memcpy(g->id, id_val, (size_t)id_len); g->id[id_len] = '\0'; }
            /* Extract xlink:href */
            const char *href_val;
            int href_len = sigil__get_attr(tag.attrs, tag.attrs_len, "xlink:href", &href_val);
            if (href_len == 0) href_len = sigil__get_attr(tag.attrs, tag.attrs_len, "href", &href_val);
            if (href_len > 0 && href_len < 127) { memcpy(g->href, href_val, (size_t)href_len); g->href[href_len] = '\0'; }
            /* gradientUnits */
            const char *gu;
            int gulen = sigil__get_attr(tag.attrs, tag.attrs_len, "gradientUnits", &gu);
            if (gulen > 0 && gulen == 16 && memcmp(gu, "userSpaceOnUse", 14) == 0) g->objectBBox = 0;
            /* spreadMethod */
            const char *sm;
            int smlen = sigil__get_attr(tag.attrs, tag.attrs_len, "spreadMethod", &sm);
            if (smlen == 7 && memcmp(sm, "reflect", 7) == 0) g->spread = 1;
            else if (smlen == 6 && memcmp(sm, "repeat", 6) == 0) g->spread = 2;
            /* Endpoint attributes */
            g->x1 = sigil__get_attr_float(tag.attrs, tag.attrs_len, "x1", g->objectBBox ? 0.0f : 0.0f);
            g->y1 = sigil__get_attr_float(tag.attrs, tag.attrs_len, "y1", 0.0f);
            g->x2 = sigil__get_attr_float(tag.attrs, tag.attrs_len, "x2", g->objectBBox ? 1.0f : 0.0f);
            g->y2 = sigil__get_attr_float(tag.attrs, tag.attrs_len, "y2", 0.0f);
            /* Handle percentage values for objectBoundingBox */
            if (g->objectBBox) {
                const char *v; int vl;
                vl = sigil__get_attr(tag.attrs, tag.attrs_len, "x1", &v);
                if (vl > 0 && v[vl-1] == '%') g->x1 = strtof(v, NULL) / 100.0f;
                vl = sigil__get_attr(tag.attrs, tag.attrs_len, "y1", &v);
                if (vl > 0 && v[vl-1] == '%') g->y1 = strtof(v, NULL) / 100.0f;
                vl = sigil__get_attr(tag.attrs, tag.attrs_len, "x2", &v);
                if (vl > 0 && v[vl-1] == '%') g->x2 = strtof(v, NULL) / 100.0f;
                vl = sigil__get_attr(tag.attrs, tag.attrs_len, "y2", &v);
                if (vl > 0 && v[vl-1] == '%') g->y2 = strtof(v, NULL) / 100.0f;
            }
            /* gradientTransform */
            const char *gt;
            int gtlen = sigil__get_attr(tag.attrs, tag.attrs_len, "gradientTransform", &gt);
            if (gtlen > 0) sigil__parse_transform(gt, gtlen, g->transform);
            in_gradient = grad_defs.count; /* 1-based index */
            if (tag.self_close) in_gradient = 0;
            continue;
        }

        if (sigil__tag_is(&tag, "radialGradient")) {
            SigilGradientDef *g = sigil__grad_array_push(&grad_defs);
            g->type = 2;
            const char *id_val;
            int id_len = sigil__get_attr(tag.attrs, tag.attrs_len, "id", &id_val);
            if (id_len > 0 && id_len < 127) { memcpy(g->id, id_val, (size_t)id_len); g->id[id_len] = '\0'; }
            const char *href_val;
            int href_len = sigil__get_attr(tag.attrs, tag.attrs_len, "xlink:href", &href_val);
            if (href_len == 0) href_len = sigil__get_attr(tag.attrs, tag.attrs_len, "href", &href_val);
            if (href_len > 0 && href_len < 127) { memcpy(g->href, href_val, (size_t)href_len); g->href[href_len] = '\0'; }
            const char *gu;
            int gulen = sigil__get_attr(tag.attrs, tag.attrs_len, "gradientUnits", &gu);
            if (gulen > 0 && gulen >= 14 && memcmp(gu, "userSpaceOnUse", 14) == 0) g->objectBBox = 0;
            const char *sm;
            int smlen = sigil__get_attr(tag.attrs, tag.attrs_len, "spreadMethod", &sm);
            if (smlen == 7 && memcmp(sm, "reflect", 7) == 0) g->spread = 1;
            else if (smlen == 6 && memcmp(sm, "repeat", 6) == 0) g->spread = 2;
            g->cx = sigil__get_attr_float(tag.attrs, tag.attrs_len, "cx", 0.5f);
            g->cy = sigil__get_attr_float(tag.attrs, tag.attrs_len, "cy", 0.5f);
            g->r  = sigil__get_attr_float(tag.attrs, tag.attrs_len, "r", 0.5f);
            g->fx = sigil__get_attr_float(tag.attrs, tag.attrs_len, "fx", -1.0f);
            g->fy = sigil__get_attr_float(tag.attrs, tag.attrs_len, "fy", -1.0f);
            g->fr = sigil__get_attr_float(tag.attrs, tag.attrs_len, "fr", 0.0f);
            if (g->objectBBox) {
                const char *v; int vl;
                vl = sigil__get_attr(tag.attrs, tag.attrs_len, "cx", &v);
                if (vl > 0 && v[vl-1] == '%') g->cx = strtof(v, NULL) / 100.0f;
                vl = sigil__get_attr(tag.attrs, tag.attrs_len, "cy", &v);
                if (vl > 0 && v[vl-1] == '%') g->cy = strtof(v, NULL) / 100.0f;
                vl = sigil__get_attr(tag.attrs, tag.attrs_len, "r", &v);
                if (vl > 0 && v[vl-1] == '%') g->r = strtof(v, NULL) / 100.0f;
            }
            /* focal defaults to center */
            if (g->fx < -0.5f) g->fx = g->cx;
            if (g->fy < -0.5f) g->fy = g->cy;
            const char *gt;
            int gtlen = sigil__get_attr(tag.attrs, tag.attrs_len, "gradientTransform", &gt);
            if (gtlen > 0) sigil__parse_transform(gt, gtlen, g->transform);
            in_gradient = grad_defs.count;
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
            /* offset */
            const char *off_val;
            int off_len = sigil__get_attr(tag.attrs, tag.attrs_len, "offset", &off_val);
            if (off_len > 0) {
                s->offset = strtof(off_val, NULL);
                if (off_val[off_len-1] == '%') s->offset /= 100.0f;
            }
            if (s->offset < 0) s->offset = 0;
            if (s->offset > 1) s->offset = 1;
            /* stop-color: check style, then attribute */
            const char *sc_style;
            int sc_slen = sigil__get_attr(tag.attrs, tag.attrs_len, "style", &sc_style);
            const char *sc_val;
            int sc_len = sigil__get_prop(tag.attrs, tag.attrs_len, sc_style, sc_slen, "stop-color", &sc_val);
            if (sc_len > 0) sigil__parse_color(sc_val, sc_len, s->color);
            /* stop-opacity */
            const char *so_val;
            int so_len = sigil__get_prop(tag.attrs, tag.attrs_len, sc_style, sc_slen, "stop-opacity", &so_val);
            if (so_len > 0) s->color[3] = strtof(so_val, NULL);
            continue;
        }

        /* Note: we don't skip shapes inside <defs> because we lack <use> support.
           Rendering them in-place is better than not rendering them at all. */

        /* <svg> tag: extract viewBox, width, height */
        if (sigil__tag_is(&tag, "svg")) {
            scene->width = sigil__get_attr_float(tag.attrs, tag.attrs_len, "width", 0);
            scene->height = sigil__get_attr_float(tag.attrs, tag.attrs_len, "height", 0);

            const char *vb;
            int vblen = sigil__get_attr(tag.attrs, tag.attrs_len, "viewBox", &vb);
            if (vblen > 0 && vb) {
                scene->has_viewBox = true;
                sscanf(vb, "%f %f %f %f",
                       &scene->viewBox[0], &scene->viewBox[1],
                       &scene->viewBox[2], &scene->viewBox[3]);
                /* Use viewBox dimensions if width/height not set */
                if (scene->width == 0) scene->width = scene->viewBox[2];
                if (scene->height == 0) scene->height = scene->viewBox[3];
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
        SigilCurve *curves = NULL;
        int curve_count = 0;
        SigilBounds shape_bounds = {0, 0, 0, 0};

        if (sigil__tag_is(&tag, "path")) {
            is_shape = 1;
            const char *d;
            int dlen = sigil__get_attr(tag.attrs, tag.attrs_len, "d", &d);
            if (dlen > 0 && d) {
                /* Need null-terminated copy */
                char *dbuf = (char *)malloc((size_t)dlen + 1);
                memcpy(dbuf, d, (size_t)dlen);
                dbuf[dlen] = '\0';
                curve_count = sigil__parse_path(dbuf, &curves, &shape_bounds);
                free(dbuf);
            }
        }
        else if (sigil__tag_is(&tag, "rect")) {
            is_shape = 1;
            float x = sigil__get_attr_float(tag.attrs, tag.attrs_len, "x", 0);
            float y = sigil__get_attr_float(tag.attrs, tag.attrs_len, "y", 0);
            float w = sigil__get_attr_float(tag.attrs, tag.attrs_len, "width", 0);
            float h = sigil__get_attr_float(tag.attrs, tag.attrs_len, "height", 0);
            float rx = sigil__get_attr_float(tag.attrs, tag.attrs_len, "rx", 0);
            float ry = sigil__get_attr_float(tag.attrs, tag.attrs_len, "ry", 0);
            if (w > 0 && h > 0) {
                curve_count = sigil__rect_to_curves(x, y, w, h, rx, ry,
                                                     &curves, &shape_bounds);
            }
        }
        else if (sigil__tag_is(&tag, "circle")) {
            is_shape = 1;
            float ccx = sigil__get_attr_float(tag.attrs, tag.attrs_len, "cx", 0);
            float ccy = sigil__get_attr_float(tag.attrs, tag.attrs_len, "cy", 0);
            float r = sigil__get_attr_float(tag.attrs, tag.attrs_len, "r", 0);
            if (r > 0) {
                curve_count = sigil__circle_to_curves(ccx, ccy, r,
                                                       &curves, &shape_bounds);
            }
        }
        else if (sigil__tag_is(&tag, "ellipse")) {
            is_shape = 1;
            float ecx = sigil__get_attr_float(tag.attrs, tag.attrs_len, "cx", 0);
            float ecy = sigil__get_attr_float(tag.attrs, tag.attrs_len, "cy", 0);
            float erx = sigil__get_attr_float(tag.attrs, tag.attrs_len, "rx", 0);
            float ery = sigil__get_attr_float(tag.attrs, tag.attrs_len, "ry", 0);
            if (erx > 0 && ery > 0) {
                curve_count = sigil__ellipse_to_curves(ecx, ecy, erx, ery,
                                                        &curves, &shape_bounds);
            }
        }
        else if (sigil__tag_is(&tag, "line")) {
            is_shape = 1;
            float lx1 = sigil__get_attr_float(tag.attrs, tag.attrs_len, "x1", 0);
            float ly1 = sigil__get_attr_float(tag.attrs, tag.attrs_len, "y1", 0);
            float lx2 = sigil__get_attr_float(tag.attrs, tag.attrs_len, "x2", 0);
            float ly2 = sigil__get_attr_float(tag.attrs, tag.attrs_len, "y2", 0);
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

        if (!is_shape) continue;

        /* Extract inline style attribute for CSS property lookups */
        const char *style_str;
        int style_len = sigil__get_attr(tag.attrs, tag.attrs_len, "style", &style_str);

        /* Check display/visibility */
        {
            const char *dv;
            int dvlen = sigil__get_prop(tag.attrs, tag.attrs_len, style_str, style_len, "display", &dv);
            if (dvlen == 4 && memcmp(dv, "none", 4) == 0) { free(curves); continue; }
            dvlen = sigil__get_prop(tag.attrs, tag.attrs_len, style_str, style_len, "visibility", &dv);
            if (dvlen == 6 && memcmp(dv, "hidden", 6) == 0) { free(curves); continue; }
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
        if (fill_vlen > 0) {
            if (fill_vlen > 4 && memcmp(fill_val, "url(", 4) == 0) {
                /* url(#id) reference — resolve gradient from local grad_defs */
                has_fill = 0; /* default to no fill if gradient not found */
                const char *hash = memchr(fill_val, '#', (size_t)fill_vlen);
                if (hash && grad_defs.count > 0) {
                    hash++;
                    const char *end_paren = memchr(hash, ')', (size_t)(fill_vlen - (int)(hash - fill_val)));
                    int id_len = end_paren ? (int)(end_paren - hash) : (int)(fill_vlen - (int)(hash - fill_val));
                    for (int gi = 0; gi < grad_defs.count; gi++) {
                        if ((int)strlen(grad_defs.data[gi].id) == id_len &&
                            memcmp(grad_defs.data[gi].id, hash, (size_t)id_len) == 0) {
                            fill_gradient_idx = gi;
                            has_fill = 1;
                            break;
                        }
                    }
                }
            } else {
                has_fill = sigil__parse_color(fill_val, fill_vlen, fill_color);
            }
        }

        const char *stroke_val;
        int stroke_vlen = sigil__get_prop(tag.attrs, tag.attrs_len, style_str, style_len, "stroke", &stroke_val);
        if (stroke_vlen > 0) {
            has_stroke = sigil__parse_color(stroke_val, stroke_vlen, stroke_color);
        }

        stroke_width = sigil__get_prop_float(tag.attrs, tag.attrs_len, style_str, style_len, "stroke-width", 0);
        if (has_stroke && stroke_width == 0) stroke_width = 1.0f; /* SVG default */

        opacity = sigil__get_prop_float(tag.attrs, tag.attrs_len, style_str, style_len, "opacity", 1.0f);
        fill_opacity = sigil__get_prop_float(tag.attrs, tag.attrs_len, style_str, style_len, "fill-opacity", 1.0f);
        stroke_opacity = sigil__get_prop_float(tag.attrs, tag.attrs_len, style_str, style_len, "stroke-opacity", 1.0f);

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

        /* stroke-miterlimit: element, then group inheritance */
        float miter_limit = 4.0f;
        {
            miter_limit = sigil__get_prop_float(tag.attrs, tag.attrs_len, style_str, style_len, "stroke-miterlimit", 0);
            if (miter_limit == 0) {
                for (int gi = xform_depth; gi >= 0; gi--) {
                    if (g_style_stack[gi]) {
                        miter_limit = sigil__get_attr_float(g_style_stack[gi], g_style_len_stack[gi], "stroke-miterlimit", 0);
                        if (miter_limit != 0) break;
                    }
                }
                if (miter_limit == 0) miter_limit = 4.0f;
            }
            if (miter_limit < 1.0f) miter_limit = 1.0f; /* SVG spec minimum */
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
            SigilCurve *stroke_curves = NULL;
            SigilBounds stroke_bounds;
            int sc = sigil__stroke_to_fill(curves, curve_count, stroke_width,
                                            line_join, line_cap, miter_limit,
                                            &stroke_curves, &stroke_bounds);
            free(curves);
            curves = stroke_curves;
            curve_count = sc;
            shape_bounds = stroke_bounds;
            /* Use stroke color as fill */
            memcpy(fill_color, stroke_color, sizeof(float) * 4);
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
            /* We need to get curves back from the fill element to offset them */
            SigilElement *fillElem = &elems.data[elems.count - 1];
            SigilCurve *stroke_curves = NULL;
            SigilBounds stroke_bounds;
            int sc = sigil__stroke_to_fill(fillElem->curves, (int)fillElem->curve_count,
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
                sigil__build_bands(se);
            }
        }

        /* Free curves if not transferred */
        free(curves);
    }

    scene->elements = elems.data;
    scene->element_count = elems.count;

    /* Resolve gradient href inheritance and store in scene */
    if (grad_defs.count > 0) {
        sigil__resolve_gradient_hrefs(&grad_defs);
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
        e[i] = (SigilSortEntry){band->curveIndices[i], mx};
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
#define STRVIEW(X) (WGPUStringView){X, sizeof(X) - 1}
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
        .chain = {.sType = WGPUSType_ShaderSourceWGSL},
        .code = {.data = vsSrc, .length = WGPU_STRLEN}
    };
    ctx->vertexShader = wgpuDeviceCreateShaderModule(device,
        &(WGPUShaderModuleDescriptor){.nextInChain = &vsWgsl.chain});

    WGPUShaderSourceWGSL fsWgsl = {
        .chain = {.sType = WGPUSType_ShaderSourceWGSL},
        .code = {.data = fsSrc, .length = WGPU_STRLEN}
    };
    ctx->fragmentShader = wgpuDeviceCreateShaderModule(device,
        &(WGPUShaderModuleDescriptor){.nextInChain = &fsWgsl.chain});

    free(vsSrc); free(fsSrc);

    /* Bind group layout: uniform + curve SSBO + band SSBO + gradient texture + sampler */
    WGPUBindGroupLayoutEntry bglEntries[5] = {
        { .binding = 0, .visibility = WGPUShaderStage_Vertex,
          .buffer = {.type = WGPUBufferBindingType_Uniform} },
        { .binding = 1, .visibility = WGPUShaderStage_Fragment,
          .buffer = {.type = WGPUBufferBindingType_ReadOnlyStorage} },
        { .binding = 2, .visibility = WGPUShaderStage_Fragment,
          .buffer = {.type = WGPUBufferBindingType_ReadOnlyStorage} },
        { .binding = 3, .visibility = WGPUShaderStage_Fragment,
          .texture = {.sampleType = WGPUTextureSampleType_Float,
                      .viewDimension = WGPUTextureViewDimension_2D} },
        { .binding = 4, .visibility = WGPUShaderStage_Fragment,
          .sampler = {.type = WGPUSamplerBindingType_Filtering} },
    };
    ctx->renderBGL = wgpuDeviceCreateBindGroupLayout(device,
        &(WGPUBindGroupLayoutDescriptor){.entryCount = 5, .entries = bglEntries});
    ctx->renderPipelineLayout = wgpuDeviceCreatePipelineLayout(device,
        &(WGPUPipelineLayoutDescriptor){.bindGroupLayoutCount = 1,
            .bindGroupLayouts = &ctx->renderBGL});

    /* Vertex attributes: 7 x Float32x4, stride = 112 */
    WGPUVertexAttribute vAttrs[7] = {
        {.shaderLocation = 0, .offset =  0, .format = WGPUVertexFormat_Float32x4},
        {.shaderLocation = 1, .offset = 16, .format = WGPUVertexFormat_Float32x4},
        {.shaderLocation = 2, .offset = 32, .format = WGPUVertexFormat_Float32x4},
        {.shaderLocation = 3, .offset = 48, .format = WGPUVertexFormat_Float32x4},
        {.shaderLocation = 4, .offset = 64, .format = WGPUVertexFormat_Float32x4},
        {.shaderLocation = 5, .offset = 80, .format = WGPUVertexFormat_Float32x4},
        {.shaderLocation = 6, .offset = 96, .format = WGPUVertexFormat_Float32x4},
    };
    WGPUVertexBufferLayout vbLayout = {
        .arrayStride = 112, .attributeCount = 7,
        .attributes = vAttrs, .stepMode = WGPUVertexStepMode_Vertex
    };

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
    };
    WGPUFragmentState fs = {
        .module = ctx->fragmentShader, .entryPoint = STRVIEW("main"),
        .targetCount = 1, .targets = &cts
    };

    WGPURenderPipelineDescriptor rpDesc = {
        .layout = ctx->renderPipelineLayout,
        .vertex = {
            .module = ctx->vertexShader, .entryPoint = STRVIEW("main"),
            .bufferCount = 1, .buffers = &vbLayout
        },
        .fragment = &fs,
        .primitive = {
            .topology = WGPUPrimitiveTopology_TriangleList,
            .cullMode = WGPUCullMode_None
        },
        .multisample = {.count = 1, .mask = 0xFFFFFFFF},
    };

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
    ctx->gradientSampler = wgpuDeviceCreateSampler(device,
        &(WGPUSamplerDescriptor){
            .magFilter = WGPUFilterMode_Linear,
            .minFilter = WGPUFilterMode_Linear,
            .addressModeU = WGPUAddressMode_ClampToEdge,
            .addressModeV = WGPUAddressMode_ClampToEdge,
            .addressModeW = WGPUAddressMode_ClampToEdge,
        });

    /* ---- Prepare compute: bind group layouts ---- */
    /* group(0): scene data inputs (read-only) */
    WGPUBindGroupLayoutEntry prepInEntries[5] = {
        { .binding = 0, .visibility = WGPUShaderStage_Compute,
          .buffer = {.type = WGPUBufferBindingType_ReadOnlyStorage} },
        { .binding = 1, .visibility = WGPUShaderStage_Compute,
          .buffer = {.type = WGPUBufferBindingType_ReadOnlyStorage} },
        { .binding = 2, .visibility = WGPUShaderStage_Compute,
          .buffer = {.type = WGPUBufferBindingType_ReadOnlyStorage} },
        { .binding = 3, .visibility = WGPUShaderStage_Compute,
          .buffer = {.type = WGPUBufferBindingType_ReadOnlyStorage} },
        { .binding = 4, .visibility = WGPUShaderStage_Compute,
          .buffer = {.type = WGPUBufferBindingType_Uniform} },
    };
    ctx->prepareInputBGL = wgpuDeviceCreateBindGroupLayout(device,
        &(WGPUBindGroupLayoutDescriptor){.entryCount = 5, .entries = prepInEntries});

    /* group(1): output buffers (read-write) */
    WGPUBindGroupLayoutEntry prepOutEntries[4] = {
        { .binding = 0, .visibility = WGPUShaderStage_Compute,
          .buffer = {.type = WGPUBufferBindingType_Storage} },
        { .binding = 1, .visibility = WGPUShaderStage_Compute,
          .buffer = {.type = WGPUBufferBindingType_Storage} },
        { .binding = 2, .visibility = WGPUShaderStage_Compute,
          .buffer = {.type = WGPUBufferBindingType_Storage} },
        { .binding = 3, .visibility = WGPUShaderStage_Compute,
          .buffer = {.type = WGPUBufferBindingType_Storage} },
    };
    ctx->prepareOutputBGL = wgpuDeviceCreateBindGroupLayout(device,
        &(WGPUBindGroupLayoutDescriptor){.entryCount = 4, .entries = prepOutEntries});

    WGPUBindGroupLayout prepareBGLs[2] = { ctx->prepareInputBGL, ctx->prepareOutputBGL };
    ctx->preparePipelineLayout = wgpuDeviceCreatePipelineLayout(device,
        &(WGPUPipelineLayoutDescriptor){.bindGroupLayoutCount = 2,
            .bindGroupLayouts = prepareBGLs});

    /* ---- Gradient compute: bind group layout ---- */
    WGPUBindGroupLayoutEntry gradEntries[3] = {
        { .binding = 0, .visibility = WGPUShaderStage_Compute,
          .buffer = {.type = WGPUBufferBindingType_ReadOnlyStorage} },
        { .binding = 1, .visibility = WGPUShaderStage_Compute,
          .buffer = {.type = WGPUBufferBindingType_ReadOnlyStorage} },
        { .binding = 2, .visibility = WGPUShaderStage_Compute,
          .buffer = {.type = WGPUBufferBindingType_Storage} },
    };
    ctx->gradientBGL = wgpuDeviceCreateBindGroupLayout(device,
        &(WGPUBindGroupLayoutDescriptor){.entryCount = 3, .entries = gradEntries});
    ctx->gradientPipelineLayout = wgpuDeviceCreatePipelineLayout(device,
        &(WGPUPipelineLayoutDescriptor){.bindGroupLayoutCount = 1,
            .bindGroupLayouts = &ctx->gradientBGL});

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
            .chain = {.sType = WGPUSType_ShaderSourceWGSL},
            .code = {.data = csPrepareSrc, .length = WGPU_STRLEN}
        };
        ctx->prepareShader = wgpuDeviceCreateShaderModule(device,
            &(WGPUShaderModuleDescriptor){.nextInChain = &csPreWgsl.chain});

        WGPUShaderSourceWGSL csGradWgsl = {
            .chain = {.sType = WGPUSType_ShaderSourceWGSL},
            .code = {.data = csGradSrc, .length = WGPU_STRLEN}
        };
        ctx->gradientShader = wgpuDeviceCreateShaderModule(device,
            &(WGPUShaderModuleDescriptor){.nextInChain = &csGradWgsl.chain});

        free(csPrepareSrc); free(csGradSrc);

        /* Create compute pipelines */
        ctx->preparePipeline = wgpuDeviceCreateComputePipeline(device,
            &(WGPUComputePipelineDescriptor){
                .layout = ctx->preparePipelineLayout,
                .compute = {.module = ctx->prepareShader, .entryPoint = STRVIEW("main")}
            });
        ctx->gradientPipeline = wgpuDeviceCreateComputePipeline(device,
            &(WGPUComputePipelineDescriptor){
                .layout = ctx->gradientPipelineLayout,
                .compute = {.module = ctx->gradientShader, .entryPoint = STRVIEW("main")}
            });
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
                grad1[2] = gr * tscale; /* transformed outer radius */
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

    WGPUDevice device = ctx->device;
    WGPUQueue queue = ctx->queue;

    /* ---- Create and upload input buffers ---- */

    /* Curves buffer */
    gs->curvesBuf = wgpuDeviceCreateBuffer(device,
        &(WGPUBufferDescriptor){
            .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
            .size = curvesBytes
        });
    wgpuQueueWriteBuffer(queue, gs->curvesBuf, 0, curvesData, curvesBytes);

    /* Elements buffer */
    gs->elementsBuf = wgpuDeviceCreateBuffer(device,
        &(WGPUBufferDescriptor){
            .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
            .size = elemBytes
        });
    wgpuQueueWriteBuffer(queue, gs->elementsBuf, 0, elemData, elemBytes);

    /* Offsets buffer */
    gs->offsetsBuf = wgpuDeviceCreateBuffer(device,
        &(WGPUBufferDescriptor){
            .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
            .size = offsetsBytes
        });
    wgpuQueueWriteBuffer(queue, gs->offsetsBuf, 0, offsetData, offsetsBytes);

    /* Gradients buffer */
    gs->gradientsBuf = wgpuDeviceCreateBuffer(device,
        &(WGPUBufferDescriptor){
            .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
            .size = gradBufBytes
        });
    if (gradBufData) {
        wgpuQueueWriteBuffer(queue, gs->gradientsBuf, 0, gradBufData, gradBufBytes);
    }

    /* Gradient stops buffer */
    gs->gradientStopsBuf = wgpuDeviceCreateBuffer(device,
        &(WGPUBufferDescriptor){
            .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
            .size = stopBufBytes
        });
    if (stopBufData) {
        wgpuQueueWriteBuffer(queue, gs->gradientStopsBuf, 0, stopBufData, stopBufBytes);
    }

    /* Viewport uniform buffer (32 bytes = 8 floats, written per-prepare) */
    gs->viewportBuf = wgpuDeviceCreateBuffer(device,
        &(WGPUBufferDescriptor){
            .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
            .size = 32
        });

    /* ---- Create output buffers ---- */
    /* All output buffers get CopyDst for CPU fallback path */

    /* curveBuf: curveOutVec4s * 16 bytes */
    uint64_t curveBufSize = curveOutVec4s > 0 ? (uint64_t)curveOutVec4s * 16 : 4;
    gs->curveBuf = wgpuDeviceCreateBuffer(device,
        &(WGPUBufferDescriptor){
            .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
            .size = curveBufSize
        });

    /* bandBuf: bandOutVec4s * 16 bytes */
    uint64_t bandBufSize = bandOutVec4s > 0 ? (uint64_t)bandOutVec4s * 16 : 4;
    gs->bandBuf = wgpuDeviceCreateBuffer(device,
        &(WGPUBufferDescriptor){
            .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
            .size = bandBufSize
        });

    /* vertexBuf: ec * 4 * 112 bytes (Vertex + CopyDst) */
    uint64_t vertexBufSize = (uint64_t)ec * 4 * 112;
    if (vertexBufSize < 4) vertexBufSize = 4;
    gs->vertexBuf = wgpuDeviceCreateBuffer(device,
        &(WGPUBufferDescriptor){
            .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc,
            .size = vertexBufSize
        });

    /* indexBuf: ec * 6 * 4 bytes (Index + CopyDst) */
    uint64_t indexBufSize = (uint64_t)ec * 6 * 4;
    if (indexBufSize < 4) indexBufSize = 4;
    gs->indexBuf = wgpuDeviceCreateBuffer(device,
        &(WGPUBufferDescriptor){
            .usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst,
            .size = indexBufSize
        });

    /* gradientRampBuf: gradTexH * 256 * 4 bytes (Storage + CopySrc) */
    uint64_t rampBufSize = (uint64_t)gradTexH * 256 * 4;
    if (rampBufSize < 4) rampBufSize = 4;
    gs->gradientRampBuf = wgpuDeviceCreateBuffer(device,
        &(WGPUBufferDescriptor){
            .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc,
            .size = rampBufSize
        });

    /* ---- Gradient ramp texture: RGBA8Unorm, 256 x gradTexH ---- */
    gs->gradientTexture = wgpuDeviceCreateTexture(device,
        &(WGPUTextureDescriptor){
            .usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
            .size = {(uint32_t)SIGIL_GRADIENT_RAMP_WIDTH, (uint32_t)gradTexH, 1},
            .format = WGPUTextureFormat_RGBA8Unorm,
            .mipLevelCount = 1, .sampleCount = 1,
            .dimension = WGPUTextureDimension_2D,
        });
    gs->gradientView = wgpuTextureCreateView(gs->gradientTexture,
        &(WGPUTextureViewDescriptor){
            .format = WGPUTextureFormat_RGBA8Unorm,
            .dimension = WGPUTextureViewDimension_2D,
            .mipLevelCount = 1, .arrayLayerCount = 1,
            .aspect = WGPUTextureAspect_All,
            .usage = WGPUTextureUsage_TextureBinding
        });

    /* ---- Build compute bind groups ---- */

    /* prepareInputBG: group(0) — 5 bindings */
    WGPUBindGroupEntry inputEntries[5] = {
        {.binding = 0, .buffer = gs->curvesBuf,    .size = curvesBytes},
        {.binding = 1, .buffer = gs->elementsBuf,  .size = elemBytes},
        {.binding = 2, .buffer = gs->offsetsBuf,   .size = offsetsBytes},
        {.binding = 3, .buffer = gs->gradientsBuf, .size = gradBufBytes},
        {.binding = 4, .buffer = gs->viewportBuf,  .size = 32},
    };
    gs->prepareInputBG = wgpuDeviceCreateBindGroup(device,
        &(WGPUBindGroupDescriptor){
            .layout = ctx->prepareInputBGL,
            .entryCount = 5, .entries = inputEntries
        });

    /* prepareOutputBG: group(1) — 4 bindings */
    WGPUBindGroupEntry outputEntries[4] = {
        {.binding = 0, .buffer = gs->curveBuf,  .size = curveBufSize},
        {.binding = 1, .buffer = gs->bandBuf,   .size = bandBufSize},
        {.binding = 2, .buffer = gs->vertexBuf, .size = vertexBufSize},
        {.binding = 3, .buffer = gs->indexBuf,  .size = indexBufSize},
    };
    gs->prepareOutputBG = wgpuDeviceCreateBindGroup(device,
        &(WGPUBindGroupDescriptor){
            .layout = ctx->prepareOutputBGL,
            .entryCount = 4, .entries = outputEntries
        });

    /* gradientBG: group(0) of gradient compute — 3 bindings (NULL if no grads) */
    if (gradCount > 0) {
        WGPUBindGroupEntry gradEntries[3] = {
            {.binding = 0, .buffer = gs->gradientsBuf,    .size = gradBufBytes},
            {.binding = 1, .buffer = gs->gradientStopsBuf, .size = stopBufBytes},
            {.binding = 2, .buffer = gs->gradientRampBuf,  .size = rampBufSize},
        };
        gs->gradientBG = wgpuDeviceCreateBindGroup(device,
            &(WGPUBindGroupDescriptor){
                .layout = ctx->gradientBGL,
                .entryCount = 3, .entries = gradEntries
            });
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
    float scale    = (vbW > 0 && vbH > 0)
                   ? (viewport_w / vbW < viewport_h / vbH ? viewport_w / vbW : viewport_h / vbH)
                   : 1.0f;
    float invScale = (scale > 0.0f) ? 1.0f / scale : 1.0f;

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

        float px0 = (xMin - vbX) * scale;
        float py0 = (yMin - vbY) * scale;
        float px1 = (xMax - vbX) * scale;
        float py1 = (yMax - vbY) * scale;

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
            vp[8] = invScale; vp[9] = 0; vp[10] = 0; vp[11] = invScale;
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

    /* ---- 4. Create SigilDrawData ---- */
    SigilDrawData *dd = (SigilDrawData *)calloc(1, sizeof *dd);
    if (!dd) return NULL;

    dd->indexCount = gs->elementCount * 6;
    dd->vertexBuf  = gs->vertexBuf;   /* borrowed, not owned */
    dd->indexBuf   = gs->indexBuf;     /* borrowed, not owned */

    /* 80-byte uniform buffer (MVP + viewport size) */
    dd->uniformBuffer = wgpuDeviceCreateBuffer(ctx->device,
        &(WGPUBufferDescriptor){
            .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
            .size  = 80
        });

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
         .size = 80},
        {.binding = 1, .buffer = gs->curveBuf,
         .size = wgpuBufferGetSize(gs->curveBuf)},
        {.binding = 2, .buffer = gs->bandBuf,
         .size = wgpuBufferGetSize(gs->bandBuf)},
        {.binding = 3, .textureView = gs->gradientView},
        {.binding = 4, .sampler = ctx->gradientSampler},
    };
    dd->renderBindGroup = wgpuDeviceCreateBindGroup(ctx->device,
        &(WGPUBindGroupDescriptor){
            .layout     = ctx->renderBGL,
            .entryCount = 5,
            .entries    = bgEntries
        });

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

    WGPURenderPassColorAttachment ca = {
        .view = color_target,
        .loadOp = clear_color ? WGPULoadOp_Clear : WGPULoadOp_Load,
        .storeOp = WGPUStoreOp_Store,
        .clearValue = clear_color
            ? (WGPUColor){clear_color[0], clear_color[1], clear_color[2], clear_color[3]}
            : (WGPUColor){0, 0, 0, 1},
        .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
    };

    WGPURenderPassDepthStencilAttachment dsa;
    WGPURenderPassDescriptor rpDesc = {
        .colorAttachmentCount = 1,
        .colorAttachments = &ca,
    };

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

#endif /* SIGIL_IMPLEMENTATION */
#endif /* SIGILVG_H */
