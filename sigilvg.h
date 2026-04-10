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

/* ------------------------------------------------------------------ */
/*  Public API                                                        */
/* ------------------------------------------------------------------ */

SigilContext* sigil_create(WGPUDevice device, WGPUTextureFormat colorFormat,
                           WGPUTextureFormat depthFormat);

SigilScene* sigil_parse_svg(const char* svg_data, size_t len);

SigilDrawData* sigil_prepare(SigilContext* ctx, SigilScene* scene,
                             float viewport_w, float viewport_h,
                             bool depth_buffering);

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

/* Bake a gradient into a 256-pixel RGBA ramp (uint8, for RGBA8Unorm texture) */
static void sigil__bake_gradient_ramp(const SigilGradientDef *g, uint8_t *ramp) {
    float color[4];
    for (int i = 0; i < SIGIL_GRADIENT_RAMP_WIDTH; i++) {
        if (g->stop_count == 0) {
            color[0] = color[1] = color[2] = color[3] = 0.0f;
        } else if (g->stop_count == 1) {
            memcpy(color, g->stops[0].color, sizeof(color));
        } else {
            float t = (float)i / (float)(SIGIL_GRADIENT_RAMP_WIDTH - 1);
            int lo = 0, hi = g->stop_count - 1;
            for (int s = 0; s < g->stop_count - 1; s++) {
                if (t >= g->stops[s].offset && t <= g->stops[s+1].offset) {
                    lo = s; hi = s + 1; break;
                }
            }
            if (t <= g->stops[0].offset) { lo = hi = 0; }
            if (t >= g->stops[g->stop_count-1].offset) { lo = hi = g->stop_count - 1; }
            float seg = g->stops[hi].offset - g->stops[lo].offset;
            float u = (seg > 1e-6f) ? (t - g->stops[lo].offset) / seg : 0.0f;
            u = u < 0.0f ? 0.0f : (u > 1.0f ? 1.0f : u);
            for (int c = 0; c < 4; c++)
                color[c] = g->stops[lo].color[c] * (1.0f - u) + g->stops[hi].color[c] * u;
        }
        for (int c = 0; c < 4; c++) {
            float v = color[c] < 0.0f ? 0.0f : (color[c] > 1.0f ? 1.0f : color[c]);
            ramp[i*4+c] = (uint8_t)(v * 255.0f + 0.5f);
        }
    }
}

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
    WGPUBuffer      vertexBuffer;
    WGPUBuffer      indexBuffer;
    WGPUBuffer      uniformBuffer;
    WGPUTexture     curveTexture;
    WGPUTexture     bandTexture;
    WGPUTextureView curveView;
    WGPUTextureView bandView;
    WGPUTexture     gradientTexture;
    WGPUTextureView gradientView;
    WGPUSampler     gradientSampler;
    WGPUBindGroup   bindGroup;
    int             indexCount;
};

struct SigilContext {
    WGPUDevice          device;
    WGPUQueue           queue;
    WGPUTextureFormat   colorFormat;
    WGPUTextureFormat   depthFormat;
    WGPURenderPipeline  pipeline;
    WGPUBindGroupLayout bindGroupLayout;
    WGPUPipelineLayout  pipelineLayout;
    WGPUShaderModule    vertexShader;
    WGPUShaderModule    fragmentShader;
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

static int sigil__stroke_to_fill(SigilCurve *curves, int count,
                                  float stroke_width,
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

    for (int i = 0; i < count; i++) {
        SigilCurve *c = &curves[i];

        /* Flatten quadratic curve to polyline */
        float *pts = NULL;
        int npts = 0, cap = 0;

        /* Add start point */
        if (npts >= cap) { cap = 64; pts = (float *)malloc((size_t)cap * 2 * sizeof(float)); }
        pts[0] = c->p0x; pts[1] = c->p0y; npts = 1;

        sigil__flatten_quad(c->p0x, c->p0y, c->p1x, c->p1y,
                            c->p2x, c->p2y, &pts, &npts, &cap);

        if (npts < 2) { free(pts); continue; }

        /* Compute normals at each point (average of adjacent segment normals) */
        float *nx = (float *)malloc((size_t)npts * sizeof(float));
        float *ny = (float *)malloc((size_t)npts * sizeof(float));

        for (int j = 0; j < npts; j++) {
            float avgNx = 0, avgNy = 0;
            int nn = 0;
            /* Previous segment normal */
            if (j > 0) {
                float sdx = pts[j*2] - pts[(j-1)*2];
                float sdy = pts[j*2+1] - pts[(j-1)*2+1];
                float slen = sqrtf(sdx*sdx + sdy*sdy);
                if (slen > 1e-12f) {
                    avgNx += -sdy / slen; avgNy += sdx / slen; nn++;
                }
            }
            /* Next segment normal */
            if (j < npts - 1) {
                float sdx = pts[(j+1)*2] - pts[j*2];
                float sdy = pts[(j+1)*2+1] - pts[j*2+1];
                float slen = sqrtf(sdx*sdx + sdy*sdy);
                if (slen > 1e-12f) {
                    avgNx += -sdy / slen; avgNy += sdx / slen; nn++;
                }
            }
            if (nn > 0) {
                float nlen = sqrtf(avgNx*avgNx + avgNy*avgNy);
                if (nlen > 1e-12f) {
                    nx[j] = avgNx / nlen;
                    ny[j] = avgNy / nlen;
                } else {
                    nx[j] = 0; ny[j] = 0;
                }
            } else {
                nx[j] = 0; ny[j] = 0;
            }
        }

        /* Build stroke outline: left side forward, right side backward */
        /* Left side: pts[j] + normal * half */
        /* Right side: pts[j] - normal * half */
        float *left  = (float *)malloc((size_t)npts * 2 * sizeof(float));
        float *right = (float *)malloc((size_t)npts * 2 * sizeof(float));
        for (int j = 0; j < npts; j++) {
            left[j*2]    = pts[j*2]   + nx[j] * half;
            left[j*2+1]  = pts[j*2+1] + ny[j] * half;
            right[j*2]   = pts[j*2]   - nx[j] * half;
            right[j*2+1] = pts[j*2+1] - ny[j] * half;
        }

        /* Emit closed contour: left[0]->left[1]->...->left[n-1]->
           right[n-1]->right[n-2]->...->right[0]->left[0] */
        /* Forward along left side */
        for (int j = 0; j < npts - 1; j++) {
            sigil__curve_array_push(&arr, sigil__line_to_quad(
                left[j*2], left[j*2+1], left[(j+1)*2], left[(j+1)*2+1]));
        }
        /* End cap: left[n-1] -> right[n-1] */
        sigil__curve_array_push(&arr, sigil__line_to_quad(
            left[(npts-1)*2], left[(npts-1)*2+1],
            right[(npts-1)*2], right[(npts-1)*2+1]));
        /* Backward along right side */
        for (int j = npts - 1; j > 0; j--) {
            sigil__curve_array_push(&arr, sigil__line_to_quad(
                right[j*2], right[j*2+1], right[(j-1)*2], right[(j-1)*2+1]));
        }
        /* Start cap: right[0] -> left[0] */
        sigil__curve_array_push(&arr, sigil__line_to_quad(
            right[0], right[1], left[0], left[1]));

        free(pts); free(nx); free(ny); free(left); free(right);
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
                                            stroke_width, &stroke_curves, &stroke_bounds);
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
/*  pack_u32_as_f32 — bit-cast uint32 to float                       */
/* ------------------------------------------------------------------ */

static float sigil__pack_u32_as_f32(uint32_t v) {
    float f;
    memcpy(&f, &v, sizeof f);
    return f;
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

    /* Load shaders — check SIGIL_SHADER_PATH env var, else current directory */
    const char *shaderDir = getenv("SIGIL_SHADER_PATH");
    char vsPath[1024], fsPath[1024];
    if (shaderDir && shaderDir[0]) {
        snprintf(vsPath, sizeof vsPath, "%s/sigil_vertex.wgsl", shaderDir);
        snprintf(fsPath, sizeof fsPath, "%s/sigil_fragment.wgsl", shaderDir);
    } else {
        snprintf(vsPath, sizeof vsPath, "sigil_vertex.wgsl");
        snprintf(fsPath, sizeof fsPath, "sigil_fragment.wgsl");
    }

    char *vsSrc = sigil__read_file(vsPath);
    char *fsSrc = sigil__read_file(fsPath);
    if (!vsSrc || !fsSrc) {
        fprintf(stderr, "sigil_create: shader files not found (%s, %s)\n", vsPath, fsPath);
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

    /* Bind group layout: uniform + curve texture + band texture + gradient texture + sampler */
    WGPUBindGroupLayoutEntry bglEntries[5] = {
        { .binding = 0, .visibility = WGPUShaderStage_Vertex,
          .buffer = {.type = WGPUBufferBindingType_Uniform} },
        { .binding = 1, .visibility = WGPUShaderStage_Fragment,
          .texture = {.sampleType = WGPUTextureSampleType_UnfilterableFloat,
                      .viewDimension = WGPUTextureViewDimension_2D} },
        { .binding = 2, .visibility = WGPUShaderStage_Fragment,
          .texture = {.sampleType = WGPUTextureSampleType_Uint,
                      .viewDimension = WGPUTextureViewDimension_2D} },
        { .binding = 3, .visibility = WGPUShaderStage_Fragment,
          .texture = {.sampleType = WGPUTextureSampleType_Float,
                      .viewDimension = WGPUTextureViewDimension_2D} },
        { .binding = 4, .visibility = WGPUShaderStage_Fragment,
          .sampler = {.type = WGPUSamplerBindingType_Filtering} },
    };
    ctx->bindGroupLayout = wgpuDeviceCreateBindGroupLayout(device,
        &(WGPUBindGroupLayoutDescriptor){.entryCount = 5, .entries = bglEntries});
    ctx->pipelineLayout = wgpuDeviceCreatePipelineLayout(device,
        &(WGPUPipelineLayoutDescriptor){.bindGroupLayoutCount = 1,
            .bindGroupLayouts = &ctx->bindGroupLayout});

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
        .layout = ctx->pipelineLayout,
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

    return ctx;
}

/* ------------------------------------------------------------------ */
/*  sigil_prepare — band building, texture packing, GPU upload        */
/* ------------------------------------------------------------------ */

SigilDrawData* sigil_prepare(SigilContext* ctx, SigilScene* scene,
                             float viewport_w, float viewport_h,
                             bool depth_buffering) {
    (void)depth_buffering;
    if (!ctx || !scene || scene->element_count == 0) return NULL;

    int ec = scene->element_count;
    SigilElement *elems = scene->elements;

    /* Sort bands for all elements */
    for (int ei = 0; ei < ec; ei++) {
        SigilElement *e = &elems[ei];
        for (int b = 0; b < SIGIL_BAND_COUNT; b++) {
            sigil__sort_band(&e->bands.hBands[b], e->curves, 0);
            sigil__sort_band(&e->bands.vBands[b], e->curves, 1);
        }
    }

    /* --- Pack curve texture (RGBA32Float, width=SIGIL_TEX_WIDTH) --- */
    const int W = SIGIL_TEX_WIDTH;

    int totalCurveTexels = 0;
    for (int ei = 0; ei < ec; ei++)
        totalCurveTexels += (int)elems[ei].curve_count * 2;

    int cH = (totalCurveTexels + W - 1) / W;
    if (cH < 1) cH = 1;
    float *curveTexData = (float *)calloc((size_t)W * (size_t)cH * 4, sizeof(float));

    /* curveStart[ei] = texel index of first curve texel for element ei */
    int *curveStart = (int *)malloc((size_t)ec * sizeof(int));
    int cIdx = 0;

    for (int ei = 0; ei < ec; ei++) {
        curveStart[ei] = cIdx;
        for (uint32_t ci = 0; ci < elems[ei].curve_count; ci++) {
            SigilCurve *c = &elems[ei].curves[ci];
            /* texel 0: p0.xy, p1.xy */
            int tx0 = cIdx % W, ty0 = cIdx / W;
            int off0 = (ty0 * W + tx0) * 4;
            curveTexData[off0]   = c->p0x; curveTexData[off0+1] = c->p0y;
            curveTexData[off0+2] = c->p1x; curveTexData[off0+3] = c->p1y;
            /* texel 1: p2.xy, unused */
            int tx1 = (cIdx+1) % W, ty1 = (cIdx+1) / W;
            int off1 = (ty1 * W + tx1) * 4;
            curveTexData[off1]   = c->p2x; curveTexData[off1+1] = c->p2y;
            cIdx += 2;
        }
    }

    /* --- Pack band texture (RGBA32Uint, width=SIGIL_TEX_WIDTH) --- */
    int totalBand = 0;
    for (int ei = 0; ei < ec; ei++) {
        int hdr = SIGIL_BAND_COUNT + SIGIL_BAND_COUNT; /* h bands + v bands */
        int pad = W - (totalBand % W);
        if (pad < hdr && pad < W) totalBand += pad;
        totalBand += hdr;
        for (int b = 0; b < SIGIL_BAND_COUNT; b++)
            totalBand += elems[ei].bands.hBands[b].count;
        for (int b = 0; b < SIGIL_BAND_COUNT; b++)
            totalBand += elems[ei].bands.vBands[b].count;
    }

    int bH = (totalBand + W - 1) / W;
    if (bH < 1) bH = 1;
    uint32_t *bandTexData = (uint32_t *)calloc((size_t)W * (size_t)bH * 4, sizeof(uint32_t));
    int *locX = (int *)malloc((size_t)ec * sizeof(int));
    int *locY = (int *)malloc((size_t)ec * sizeof(int));
    int bIdx = 0;

    for (int ei = 0; ei < ec; ei++) {
        SigilElement *e = &elems[ei];
        int hbc = SIGIL_BAND_COUNT, vbc = SIGIL_BAND_COUNT;
        int hdr = hbc + vbc;

        /* avoid header row-wrap */
        int cx = bIdx % W;
        if (cx + hdr > W) bIdx = (bIdx / W + 1) * W;

        locX[ei] = bIdx % W;
        locY[ei] = bIdx / W;
        int elemStart = bIdx;
        int gcs = curveStart[ei];

        /* compute offsets for curve lists */
        int clOff = hdr;
        int *offs = (int *)malloc((size_t)hdr * sizeof(int));
        for (int i = 0; i < hbc; i++) {
            offs[i] = clOff;
            clOff += e->bands.hBands[i].count;
        }
        for (int i = 0; i < vbc; i++) {
            offs[hbc + i] = clOff;
            clOff += e->bands.vBands[i].count;
        }

        /* write headers */
        for (int i = 0; i < hdr; i++) {
            int tl = elemStart + i;
            int di = (tl / W * W + tl % W) * 4;
            int cnt = i < hbc ? e->bands.hBands[i].count
                              : e->bands.vBands[i - hbc].count;
            bandTexData[di]   = (uint32_t)cnt;
            bandTexData[di+1] = (uint32_t)offs[i];
        }

        /* write curve lists */
        for (int i = 0; i < hdr; i++) {
            SigilBandEntry *band = i < hbc ? &e->bands.hBands[i]
                                           : &e->bands.vBands[i - hbc];
            int ls = elemStart + offs[i];
            for (int j = 0; j < band->count; j++) {
                int ct = gcs + band->curveIndices[j] * 2;
                int tl = ls + j;
                int di = (tl / W * W + tl % W) * 4;
                bandTexData[di]   = (uint32_t)(ct % W);
                bandTexData[di+1] = (uint32_t)(ct / W);
            }
        }

        free(offs);
        bIdx = elemStart + clOff;
    }

    /* --- Build vertex/index buffers --- */
    /* viewBox -> viewport mapping */
    float vbX = 0, vbY = 0, vbW = viewport_w, vbH = viewport_h;
    if (scene->has_viewBox) {
        vbX = scene->viewBox[0];
        vbY = scene->viewBox[1];
        vbW = scene->viewBox[2];
        vbH = scene->viewBox[3];
    }
    float scale = fminf(viewport_w / vbW, viewport_h / vbH);
    float invScale = 1.0f / scale;

    /* --- Bake gradient ramp textures (RGBA8Unorm for filterable sampling) --- */
    int gradCount = scene->gradient_count;
    int gradTexH = gradCount > 0 ? gradCount : 1;
    uint8_t *gradTexData = (uint8_t *)calloc((size_t)SIGIL_GRADIENT_RAMP_WIDTH * (size_t)gradTexH * 4, 1);
    for (int gi = 0; gi < gradCount; gi++) {
        sigil__bake_gradient_ramp(&scene->gradients[gi],
            gradTexData + gi * SIGIL_GRADIENT_RAMP_WIDTH * 4);
    }

    /* 28 floats per vertex (7 x float4), stride = 112 */
    float *verts = (float *)malloc((size_t)ec * 4 * 28 * sizeof(float));
    uint32_t *idxs = (uint32_t *)malloc((size_t)ec * 6 * sizeof(uint32_t));
    int vc = 0, ic = 0, qi = 0;

    for (int ei = 0; ei < ec; ei++) {
        SigilElement *e = &elems[ei];
        if (e->curve_count == 0) continue;

        float xMin = e->bounds.xMin, yMin = e->bounds.yMin;
        float xMax = e->bounds.xMax, yMax = e->bounds.yMax;
        float ew = xMax - xMin, eh = yMax - yMin;

        /* Convert from viewBox coords to pixel coords */
        float px0 = (xMin - vbX) * scale;
        float py0 = (yMin - vbY) * scale;
        float px1 = (xMax - vbX) * scale;
        float py1 = (yMax - vbY) * scale;

        /* band scale/offset in em-space (viewBox coords) */
        float bsX = ew > 0 ? (float)SIGIL_BAND_COUNT / ew : 0;
        float bsY = eh > 0 ? (float)SIGIL_BAND_COUNT / eh : 0;
        float boX = -xMin * bsX;
        float boY = -yMin * bsY;

        /* Pack glyph location and band maxes */
        float glp = sigil__pack_u32_as_f32(
            ((uint32_t)locY[ei] << 16) | (uint32_t)locX[ei]);

        uint32_t bandMaxPacked = ((uint32_t)(SIGIL_BAND_COUNT - 1) << 16)
                               | (uint32_t)(SIGIL_BAND_COUNT - 1);
        /* Set even-odd flag at bit 28 if needed */
        if (e->fill_rule == SIGIL_FILL_EVENODD)
            bandMaxPacked |= (1u << 28);
        float bmp = sigil__pack_u32_as_f32(bandMaxPacked);

        /* Color: premultiplied alpha */
        float alpha = e->fill_color[3] * e->opacity;
        float cr = e->fill_color[0] * alpha;
        float cg = e->fill_color[1] * alpha;
        float cb = e->fill_color[2] * alpha;

        /* Gradient params (attrs 5+6): default to zeros (solid fill) */
        float grad0[4] = {0, 0, 0, 0};
        float grad1[4] = {0, 0, 0, 0};
        int isGradient = 0;

        if (e->fill_gradient_idx >= 0 && e->fill_gradient_idx < gradCount) {
            const SigilGradientDef *gd = &scene->gradients[e->fill_gradient_idx];
            isGradient = 1;
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

            /* For gradient elements: set color to white with element opacity
               (gradient color comes from texture sampling in fragment shader) */
            cr = 0; cg = 0; cb = 0; alpha = -e->opacity; /* negative alpha = gradient signal */
        }

        /* 4 corners: pixel-space position, normal direction, em-space coords */
        float corners[4][6] = {
            {px0, py0, -1, -1, xMin, yMin},
            {px1, py0,  1, -1, xMax, yMin},
            {px1, py1,  1,  1, xMax, yMax},
            {px0, py1, -1,  1, xMin, yMax},
        };

        for (int c = 0; c < 4; c++) {
            float *v = &verts[vc * 28];
            /* attr 0: position.xy, normal.xy */
            v[ 0] = corners[c][0]; v[ 1] = corners[c][1];
            v[ 2] = corners[c][2]; v[ 3] = corners[c][3];
            /* attr 1: glyphCoord.xy, glyphLoc, bandMaxes */
            v[ 4] = corners[c][4]; v[ 5] = corners[c][5];
            v[ 6] = glp;           v[ 7] = bmp;
            /* attr 2: jacobian (invScale, 0, 0, invScale) */
            v[ 8] = invScale;      v[ 9] = 0;
            v[10] = 0;             v[11] = invScale;
            /* attr 3: bandScale.xy, bandOffset.xy */
            v[12] = bsX;           v[13] = bsY;
            v[14] = boX;           v[15] = boY;
            /* attr 4: color.rgba (premultiplied; negative alpha = gradient) */
            v[16] = cr;            v[17] = cg;
            v[18] = cb;            v[19] = alpha;
            /* attr 5: grad0 (linear: x1,y1,x2,y2; radial: cx,cy,fx,fy) */
            v[20] = grad0[0];     v[21] = grad0[1];
            v[22] = grad0[2];     v[23] = grad0[3];
            /* attr 6: grad1 (gradTexRow, gradType, radius, spread) */
            v[24] = grad1[0];     v[25] = grad1[1];
            v[26] = grad1[2];     v[27] = grad1[3];
            vc++;
        }

        uint32_t base = (uint32_t)(qi * 4);
        idxs[ic++] = base;     idxs[ic++] = base + 1; idxs[ic++] = base + 2;
        idxs[ic++] = base;     idxs[ic++] = base + 2; idxs[ic++] = base + 3;
        qi++;
    }

    free(curveStart);

    if (vc == 0 || ic == 0) {
        free(verts); free(idxs); free(locX); free(locY);
        free(curveTexData); free(bandTexData); free(gradTexData);
        return NULL;
    }

    /* --- Upload to GPU --- */
    SigilDrawData *dd = (SigilDrawData *)calloc(1, sizeof(SigilDrawData));
    dd->indexCount = ic;

    WGPUQueue queue = ctx->queue;

    /* Vertex buffer */
    dd->vertexBuffer = wgpuDeviceCreateBuffer(ctx->device,
        &(WGPUBufferDescriptor){
            .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
            .size = (uint64_t)vc * 112
        });
    wgpuQueueWriteBuffer(queue, dd->vertexBuffer, 0, verts, (size_t)vc * 112);

    /* Index buffer */
    dd->indexBuffer = wgpuDeviceCreateBuffer(ctx->device,
        &(WGPUBufferDescriptor){
            .usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst,
            .size = (uint64_t)ic * 4
        });
    wgpuQueueWriteBuffer(queue, dd->indexBuffer, 0, idxs, (size_t)ic * 4);

    /* Uniform buffer (80 bytes = 20 floats) */
    dd->uniformBuffer = wgpuDeviceCreateBuffer(ctx->device,
        &(WGPUBufferDescriptor){
            .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
            .size = 80
        });

    /* MVP matrix: pixel coords → clip space. SVG Y-down → WebGPU Y-up. */
    float ubo[20] = {
         2.0f / viewport_w, 0, 0, -1.0f,
         0, -2.0f / viewport_h, 0,  1.0f,  /* negate Y for SVG Y-down */
         0, 0, 0, 0,
         0, 0, 0, 1,
         viewport_w, viewport_h, 0, 0,
    };
    wgpuQueueWriteBuffer(queue, dd->uniformBuffer, 0, ubo, sizeof ubo);

    /* Curve texture (RGBA32Float) */
    dd->curveTexture = wgpuDeviceCreateTexture(ctx->device,
        &(WGPUTextureDescriptor){
            .usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
            .size = {(uint32_t)W, (uint32_t)cH, 1},
            .format = WGPUTextureFormat_RGBA32Float,
            .mipLevelCount = 1, .sampleCount = 1,
            .dimension = WGPUTextureDimension_2D,
        });
    wgpuQueueWriteTexture(queue,
        &(WGPUTexelCopyTextureInfo){.texture = dd->curveTexture, .aspect = WGPUTextureAspect_All},
        curveTexData, (size_t)W * (size_t)cH * 16,
        &(WGPUTexelCopyBufferLayout){.bytesPerRow = (uint32_t)(W * 16), .rowsPerImage = (uint32_t)cH},
        &(WGPUExtent3D){(uint32_t)W, (uint32_t)cH, 1});
    dd->curveView = wgpuTextureCreateView(dd->curveTexture,
        &(WGPUTextureViewDescriptor){
            .format = WGPUTextureFormat_RGBA32Float,
            .dimension = WGPUTextureViewDimension_2D,
            .mipLevelCount = 1, .arrayLayerCount = 1,
            .aspect = WGPUTextureAspect_All,
            .usage = WGPUTextureUsage_TextureBinding
        });

    /* Band texture (RGBA32Uint) */
    dd->bandTexture = wgpuDeviceCreateTexture(ctx->device,
        &(WGPUTextureDescriptor){
            .usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
            .size = {(uint32_t)W, (uint32_t)bH, 1},
            .format = WGPUTextureFormat_RGBA32Uint,
            .mipLevelCount = 1, .sampleCount = 1,
            .dimension = WGPUTextureDimension_2D,
        });
    wgpuQueueWriteTexture(queue,
        &(WGPUTexelCopyTextureInfo){.texture = dd->bandTexture, .aspect = WGPUTextureAspect_All},
        bandTexData, (size_t)W * (size_t)bH * 16,
        &(WGPUTexelCopyBufferLayout){.bytesPerRow = (uint32_t)(W * 16), .rowsPerImage = (uint32_t)bH},
        &(WGPUExtent3D){(uint32_t)W, (uint32_t)bH, 1});
    dd->bandView = wgpuTextureCreateView(dd->bandTexture,
        &(WGPUTextureViewDescriptor){
            .format = WGPUTextureFormat_RGBA32Uint,
            .dimension = WGPUTextureViewDimension_2D,
            .mipLevelCount = 1, .arrayLayerCount = 1,
            .aspect = WGPUTextureAspect_All,
            .usage = WGPUTextureUsage_TextureBinding
        });

    /* Gradient ramp texture (RGBA8Unorm — filterable for linear sampling) */
    dd->gradientTexture = wgpuDeviceCreateTexture(ctx->device,
        &(WGPUTextureDescriptor){
            .usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
            .size = {(uint32_t)SIGIL_GRADIENT_RAMP_WIDTH, (uint32_t)gradTexH, 1},
            .format = WGPUTextureFormat_RGBA8Unorm,
            .mipLevelCount = 1, .sampleCount = 1,
            .dimension = WGPUTextureDimension_2D,
        });
    wgpuQueueWriteTexture(queue,
        &(WGPUTexelCopyTextureInfo){.texture = dd->gradientTexture, .aspect = WGPUTextureAspect_All},
        gradTexData, (size_t)SIGIL_GRADIENT_RAMP_WIDTH * (size_t)gradTexH * 4,
        &(WGPUTexelCopyBufferLayout){.bytesPerRow = (uint32_t)(SIGIL_GRADIENT_RAMP_WIDTH * 4),
            .rowsPerImage = (uint32_t)gradTexH},
        &(WGPUExtent3D){(uint32_t)SIGIL_GRADIENT_RAMP_WIDTH, (uint32_t)gradTexH, 1});
    dd->gradientView = wgpuTextureCreateView(dd->gradientTexture,
        &(WGPUTextureViewDescriptor){
            .format = WGPUTextureFormat_RGBA8Unorm,
            .dimension = WGPUTextureViewDimension_2D,
            .mipLevelCount = 1, .arrayLayerCount = 1,
            .aspect = WGPUTextureAspect_All,
            .usage = WGPUTextureUsage_TextureBinding
        });
    dd->gradientSampler = wgpuDeviceCreateSampler(ctx->device,
        &(WGPUSamplerDescriptor){
            .magFilter = WGPUFilterMode_Linear,
            .minFilter = WGPUFilterMode_Linear,
            .addressModeU = WGPUAddressMode_ClampToEdge,
            .addressModeV = WGPUAddressMode_ClampToEdge,
        });

    /* Bind group */
    WGPUBindGroupEntry bgEntries[5] = {
        {.binding = 0, .buffer = dd->uniformBuffer, .size = 80},
        {.binding = 1, .textureView = dd->curveView},
        {.binding = 2, .textureView = dd->bandView},
        {.binding = 3, .textureView = dd->gradientView},
        {.binding = 4, .sampler = dd->gradientSampler},
    };
    dd->bindGroup = wgpuDeviceCreateBindGroup(ctx->device,
        &(WGPUBindGroupDescriptor){
            .layout = ctx->bindGroupLayout,
            .entryCount = 5, .entries = bgEntries
        });

    /* Free CPU-side data */
    free(verts); free(idxs); free(locX); free(locY);
    free(curveTexData); free(bandTexData); free(gradTexData);

    return dd;
}

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
    wgpuRenderPassEncoderSetBindGroup(rp, 0, data->bindGroup, 0, NULL);
    wgpuRenderPassEncoderSetVertexBuffer(rp, 0, data->vertexBuffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(rp, data->indexBuffer, WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDrawIndexed(rp, (uint32_t)data->indexCount, 1, 0, 0, 0);
    wgpuRenderPassEncoderEnd(rp);
    wgpuRenderPassEncoderRelease(rp);
}

/* ------------------------------------------------------------------ */
/*  sigil_free_draw_data — release GPU resources                      */
/* ------------------------------------------------------------------ */

void sigil_free_draw_data(SigilDrawData* data) {
    if (!data) return;
    if (data->bindGroup)        wgpuBindGroupRelease(data->bindGroup);
    if (data->curveView)        wgpuTextureViewRelease(data->curveView);
    if (data->bandView)         wgpuTextureViewRelease(data->bandView);
    if (data->gradientView)     wgpuTextureViewRelease(data->gradientView);
    if (data->gradientSampler)  wgpuSamplerRelease(data->gradientSampler);
    if (data->curveTexture)     wgpuTextureDestroy(data->curveTexture);
    if (data->curveTexture)     wgpuTextureRelease(data->curveTexture);
    if (data->bandTexture)      wgpuTextureDestroy(data->bandTexture);
    if (data->bandTexture)      wgpuTextureRelease(data->bandTexture);
    if (data->gradientTexture)  wgpuTextureDestroy(data->gradientTexture);
    if (data->gradientTexture)  wgpuTextureRelease(data->gradientTexture);
    if (data->vertexBuffer)     wgpuBufferDestroy(data->vertexBuffer);
    if (data->vertexBuffer)     wgpuBufferRelease(data->vertexBuffer);
    if (data->indexBuffer)      wgpuBufferDestroy(data->indexBuffer);
    if (data->indexBuffer)      wgpuBufferRelease(data->indexBuffer);
    if (data->uniformBuffer)    wgpuBufferDestroy(data->uniformBuffer);
    if (data->uniformBuffer)    wgpuBufferRelease(data->uniformBuffer);
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
    if (ctx->pipeline)        wgpuRenderPipelineRelease(ctx->pipeline);
    if (ctx->pipelineLayout)  wgpuPipelineLayoutRelease(ctx->pipelineLayout);
    if (ctx->bindGroupLayout) wgpuBindGroupLayoutRelease(ctx->bindGroupLayout);
    if (ctx->vertexShader)    wgpuShaderModuleRelease(ctx->vertexShader);
    if (ctx->fragmentShader)  wgpuShaderModuleRelease(ctx->fragmentShader);
    free(ctx);
}

#endif /* SIGIL_IMPLEMENTATION */
#endif /* SIGILVG_H */
