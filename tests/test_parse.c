#define SIGIL_IMPLEMENTATION
#include "sigilvg.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

static int g_pass = 0, g_fail = 0;

#define TEST(name) static void name(void)
#define ASSERT(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        g_fail++; return; \
    } \
} while(0)
#define ASSERT_NEAR(a, b, eps) ASSERT(fabsf((float)(a)-(float)(b)) < (eps))
#define RUN(name) do { \
    int _prev_fail = g_fail; \
    printf("  %s...", #name); name(); \
    if (g_fail == _prev_fail) { printf(" ok\n"); g_pass++; } \
    else { printf(" FAILED\n"); } \
} while(0)

/* ================================================================== */
/*  Path parser tests                                                 */
/* ================================================================== */

TEST(test_parse_moveto_lineto) {
    SigilCurve *curves = NULL;
    SigilBounds bounds;
    int n = sigil__parse_path("M 0 0 L 100 0 L 50 100 Z", &curves, &bounds);
    ASSERT(n == 3);
    /* First curve: (0,0) -> (100,0) */
    ASSERT_NEAR(curves[0].p0x, 0.0f, 0.1f);
    ASSERT_NEAR(curves[0].p0y, 0.0f, 0.1f);
    ASSERT_NEAR(curves[0].p2x, 100.0f, 0.1f);
    ASSERT_NEAR(curves[0].p2y, 0.0f, 0.1f);
    /* Bounds */
    ASSERT_NEAR(bounds.xMin, 0.0f, 1.0f);
    ASSERT_NEAR(bounds.yMin, 0.0f, 1.0f);
    ASSERT(bounds.xMax >= 99.0f);
    ASSERT(bounds.yMax >= 99.0f);
    free(curves);
}

TEST(test_parse_relative_lineto) {
    SigilCurve *curves = NULL;
    SigilBounds bounds;
    int n = sigil__parse_path("M 10 10 l 50 0 l 0 50 z", &curves, &bounds);
    ASSERT(n == 3);
    /* First line: (10,10) -> (60,10) */
    ASSERT_NEAR(curves[0].p0x, 10.0f, 0.1f);
    ASSERT_NEAR(curves[0].p0y, 10.0f, 0.1f);
    ASSERT_NEAR(curves[0].p2x, 60.0f, 0.1f);
    ASSERT_NEAR(curves[0].p2y, 10.0f, 0.1f);
    /* Second line: (60,10) -> (60,60) */
    ASSERT_NEAR(curves[1].p0x, 60.0f, 0.1f);
    ASSERT_NEAR(curves[1].p0y, 10.0f, 0.1f);
    ASSERT_NEAR(curves[1].p2x, 60.0f, 0.1f);
    ASSERT_NEAR(curves[1].p2y, 60.0f, 0.1f);
    /* Third line (close): (60,60) -> (10,10) */
    ASSERT_NEAR(curves[2].p0x, 60.0f, 0.1f);
    ASSERT_NEAR(curves[2].p0y, 60.0f, 0.1f);
    ASSERT_NEAR(curves[2].p2x, 10.0f, 0.1f);
    ASSERT_NEAR(curves[2].p2y, 10.0f, 0.1f);
    free(curves);
}

TEST(test_parse_hv_lines) {
    SigilCurve *curves = NULL;
    SigilBounds bounds;
    int n = sigil__parse_path("M 0 0 H 100 V 100 H 0 Z", &curves, &bounds);
    ASSERT(n == 4);
    /* H 100: (0,0) -> (100,0) */
    ASSERT_NEAR(curves[0].p2x, 100.0f, 0.1f);
    ASSERT_NEAR(curves[0].p2y, 0.0f, 0.1f);
    /* V 100: (100,0) -> (100,100) */
    ASSERT_NEAR(curves[1].p2x, 100.0f, 0.1f);
    ASSERT_NEAR(curves[1].p2y, 100.0f, 0.1f);
    /* H 0: (100,100) -> (0,100) */
    ASSERT_NEAR(curves[2].p2x, 0.0f, 0.1f);
    ASSERT_NEAR(curves[2].p2y, 100.0f, 0.1f);
    /* Z: (0,100) -> (0,0) */
    ASSERT_NEAR(curves[3].p2x, 0.0f, 0.1f);
    ASSERT_NEAR(curves[3].p2y, 0.0f, 0.1f);
    free(curves);
}

TEST(test_parse_cubic) {
    SigilCurve *curves = NULL;
    SigilBounds bounds;
    int n = sigil__parse_path("M 0 0 C 10 20 30 40 50 60", &curves, &bounds);
    /* Cubic splits into 2 quadratics via de Casteljau */
    ASSERT(n == 2);
    /* First quad starts at (0,0) */
    ASSERT_NEAR(curves[0].p0x, 0.0f, 0.1f);
    ASSERT_NEAR(curves[0].p0y, 0.0f, 0.1f);
    /* Second quad ends at (50,60) */
    ASSERT_NEAR(curves[1].p2x, 50.0f, 0.1f);
    ASSERT_NEAR(curves[1].p2y, 60.0f, 0.1f);
    /* Midpoint should be connected */
    ASSERT_NEAR(curves[0].p2x, curves[1].p0x, 0.01f);
    ASSERT_NEAR(curves[0].p2y, curves[1].p0y, 0.01f);
    free(curves);
}

TEST(test_parse_quadratic) {
    SigilCurve *curves = NULL;
    SigilBounds bounds;
    int n = sigil__parse_path("M 0 0 Q 50 100 100 0", &curves, &bounds);
    ASSERT(n == 1);
    /* Exact control points for quadratic */
    ASSERT_NEAR(curves[0].p0x, 0.0f, 0.01f);
    ASSERT_NEAR(curves[0].p0y, 0.0f, 0.01f);
    ASSERT_NEAR(curves[0].p1x, 50.0f, 0.01f);
    ASSERT_NEAR(curves[0].p1y, 100.0f, 0.01f);
    ASSERT_NEAR(curves[0].p2x, 100.0f, 0.01f);
    ASSERT_NEAR(curves[0].p2y, 0.0f, 0.01f);
    free(curves);
}

TEST(test_parse_smooth_cubic) {
    SigilCurve *curves = NULL;
    SigilBounds bounds;
    int n = sigil__parse_path("M 0 0 C 10 20 30 40 50 60 S 70 80 90 100",
                              &curves, &bounds);
    /* First cubic -> 2 quads, S (smooth cubic) -> 2 more quads = 4 total */
    ASSERT(n >= 3);
    /* Last curve ends at (90,100) */
    ASSERT_NEAR(curves[n-1].p2x, 90.0f, 0.1f);
    ASSERT_NEAR(curves[n-1].p2y, 100.0f, 0.1f);
    free(curves);
}

TEST(test_parse_arc) {
    SigilCurve *curves = NULL;
    SigilBounds bounds;
    int n = sigil__parse_path("M 10 80 A 45 45 0 0 0 125 125", &curves, &bounds);
    ASSERT(n >= 1);
    /* First curve starts near (10,80) */
    ASSERT_NEAR(curves[0].p0x, 10.0f, 0.5f);
    ASSERT_NEAR(curves[0].p0y, 80.0f, 0.5f);
    /* Last curve ends near (125,125) */
    ASSERT_NEAR(curves[n-1].p2x, 125.0f, 1.0f);
    ASSERT_NEAR(curves[n-1].p2y, 125.0f, 1.0f);
    free(curves);
}

TEST(test_parse_comma_separated) {
    SigilCurve *curves = NULL;
    SigilBounds bounds;
    int n = sigil__parse_path("M0,0L100,0L50,100Z", &curves, &bounds);
    ASSERT(n == 3);
    ASSERT_NEAR(curves[0].p0x, 0.0f, 0.1f);
    ASSERT_NEAR(curves[0].p2x, 100.0f, 0.1f);
    ASSERT_NEAR(curves[1].p2x, 50.0f, 0.1f);
    ASSERT_NEAR(curves[1].p2y, 100.0f, 0.1f);
    free(curves);
}

/* ================================================================== */
/*  SVG parsing tests                                                 */
/* ================================================================== */

TEST(test_parse_svg_rect) {
    const char *svg =
        "<svg width=\"200\" height=\"200\">"
        "  <rect x=\"10\" y=\"10\" width=\"80\" height=\"60\" fill=\"red\"/>"
        "</svg>";
    SigilScene *scene = sigil_parse_svg(svg, strlen(svg));
    ASSERT(scene != NULL);
    ASSERT(scene->element_count == 1);
    /* Red fill = (1,0,0,1) */
    ASSERT_NEAR(scene->elements[0].fill_color[0], 1.0f, 0.01f);
    ASSERT_NEAR(scene->elements[0].fill_color[1], 0.0f, 0.01f);
    ASSERT_NEAR(scene->elements[0].fill_color[2], 0.0f, 0.01f);
    ASSERT(scene->elements[0].curve_count > 0);
    sigil_free_scene(scene);
}

TEST(test_parse_svg_circle) {
    const char *svg =
        "<svg width=\"200\" height=\"200\">"
        "  <circle cx=\"100\" cy=\"100\" r=\"50\" fill=\"blue\"/>"
        "</svg>";
    SigilScene *scene = sigil_parse_svg(svg, strlen(svg));
    ASSERT(scene != NULL);
    ASSERT(scene->element_count == 1);
    /* Blue fill = (0,0,1,1) */
    ASSERT_NEAR(scene->elements[0].fill_color[0], 0.0f, 0.01f);
    ASSERT_NEAR(scene->elements[0].fill_color[1], 0.0f, 0.01f);
    ASSERT_NEAR(scene->elements[0].fill_color[2], 1.0f, 0.01f);
    ASSERT(scene->elements[0].curve_count > 0);
    sigil_free_scene(scene);
}

TEST(test_parse_svg_path) {
    const char *svg =
        "<svg width=\"200\" height=\"200\">"
        "  <path d=\"M 0 0 L 100 0 L 50 100 Z\" fill=\"#00ff00\"/>"
        "</svg>";
    SigilScene *scene = sigil_parse_svg(svg, strlen(svg));
    ASSERT(scene != NULL);
    ASSERT(scene->element_count == 1);
    /* Green fill = (0,1,0,1) */
    ASSERT_NEAR(scene->elements[0].fill_color[0], 0.0f, 0.01f);
    ASSERT_NEAR(scene->elements[0].fill_color[1], 1.0f, 0.01f);
    ASSERT_NEAR(scene->elements[0].fill_color[2], 0.0f, 0.01f);
    ASSERT(scene->elements[0].curve_count == 3);
    sigil_free_scene(scene);
}

TEST(test_parse_svg_group_transform) {
    const char *svg =
        "<svg width=\"200\" height=\"200\">"
        "  <g transform=\"translate(10,20)\">"
        "    <rect x=\"0\" y=\"0\" width=\"50\" height=\"50\" fill=\"black\"/>"
        "  </g>"
        "</svg>";
    SigilScene *scene = sigil_parse_svg(svg, strlen(svg));
    ASSERT(scene != NULL);
    ASSERT(scene->element_count == 1);
    /* The rect (0,0)-(50,50) should be translated by (10,20) */
    /* So the first curve's p0 should be near (10,20) */
    ASSERT_NEAR(scene->elements[0].transform[4], 10.0f, 0.01f);
    ASSERT_NEAR(scene->elements[0].transform[5], 20.0f, 0.01f);
    /* Bounds should reflect the translation */
    ASSERT(scene->elements[0].bounds.xMin >= 9.5f);
    ASSERT(scene->elements[0].bounds.yMin >= 19.5f);
    sigil_free_scene(scene);
}

TEST(test_parse_svg_multiple_elements) {
    const char *svg =
        "<svg width=\"200\" height=\"200\">"
        "  <rect x=\"0\" y=\"0\" width=\"50\" height=\"50\" fill=\"red\"/>"
        "  <circle cx=\"100\" cy=\"100\" r=\"30\" fill=\"blue\"/>"
        "  <path d=\"M 0 0 L 100 100\" fill=\"green\"/>"
        "</svg>";
    SigilScene *scene = sigil_parse_svg(svg, strlen(svg));
    ASSERT(scene != NULL);
    ASSERT(scene->element_count == 3);
    sigil_free_scene(scene);
}

TEST(test_parse_svg_ellipse) {
    const char *svg =
        "<svg width=\"200\" height=\"200\">"
        "  <ellipse cx=\"100\" cy=\"100\" rx=\"80\" ry=\"40\" fill=\"orange\"/>"
        "</svg>";
    SigilScene *scene = sigil_parse_svg(svg, strlen(svg));
    ASSERT(scene != NULL);
    ASSERT(scene->element_count == 1);
    ASSERT(scene->elements[0].curve_count > 0);
    sigil_free_scene(scene);
}

TEST(test_parse_svg_polyline) {
    const char *svg =
        "<svg width=\"200\" height=\"200\">"
        "  <polyline points=\"10,10 50,50 90,10\" fill=\"none\" stroke=\"black\" stroke-width=\"2\"/>"
        "</svg>";
    SigilScene *scene = sigil_parse_svg(svg, strlen(svg));
    ASSERT(scene != NULL);
    ASSERT(scene->element_count >= 1);
    sigil_free_scene(scene);
}

TEST(test_parse_color_names) {
    const char *svg =
        "<svg width=\"200\" height=\"200\">"
        "  <rect x=\"0\" y=\"0\" width=\"50\" height=\"50\" fill=\"navy\"/>"
        "</svg>";
    SigilScene *scene = sigil_parse_svg(svg, strlen(svg));
    ASSERT(scene != NULL);
    ASSERT(scene->element_count == 1);
    /* navy = (0, 0, 0.502) */
    ASSERT_NEAR(scene->elements[0].fill_color[0], 0.0f, 0.01f);
    ASSERT_NEAR(scene->elements[0].fill_color[1], 0.0f, 0.01f);
    ASSERT_NEAR(scene->elements[0].fill_color[2], 0.502f, 0.01f);
    sigil_free_scene(scene);
}

TEST(test_parse_fill_rule) {
    const char *svg =
        "<svg width=\"200\" height=\"200\">"
        "  <path d=\"M 0 0 L 100 0 L 50 100 Z\" fill=\"black\" fill-rule=\"evenodd\"/>"
        "</svg>";
    SigilScene *scene = sigil_parse_svg(svg, strlen(svg));
    ASSERT(scene != NULL);
    ASSERT(scene->element_count == 1);
    ASSERT(scene->elements[0].fill_rule == SIGIL_FILL_EVENODD);
    sigil_free_scene(scene);
}

TEST(test_stroke_generates_fill) {
    const char *svg =
        "<svg width=\"200\" height=\"200\">"
        "  <line x1=\"10\" y1=\"10\" x2=\"100\" y2=\"100\" fill=\"none\" stroke=\"red\" stroke-width=\"4\"/>"
        "</svg>";
    SigilScene *scene = sigil_parse_svg(svg, strlen(svg));
    ASSERT(scene != NULL);
    ASSERT(scene->element_count >= 1);
    /* Stroke was converted to fill, so fill_color should be red */
    ASSERT_NEAR(scene->elements[0].fill_color[0], 1.0f, 0.01f);
    ASSERT_NEAR(scene->elements[0].fill_color[1], 0.0f, 0.01f);
    ASSERT_NEAR(scene->elements[0].fill_color[2], 0.0f, 0.01f);
    ASSERT(scene->elements[0].curve_count > 0);
    sigil_free_scene(scene);
}

/* ================================================================== */
/*  Stroke join/cap tests                                             */
/* ================================================================== */

TEST(test_stroke_miter_join) {
    /* V-shape: miter join should extend above the vertex */
    const char *svg =
        "<svg viewBox=\"0 0 200 200\">"
        "  <path d=\"M 20 170 L 100 30 L 180 170\" fill=\"none\" stroke=\"green\" stroke-width=\"20\"/>"
        "</svg>";
    SigilScene *scene = sigil_parse_svg(svg, strlen(svg));
    ASSERT(scene != NULL);
    ASSERT(scene->element_count == 1);
    ASSERT(scene->elements[0].curve_count > 0);
    /* Miter join should extend above y=30 (the vertex) */
    ASSERT(scene->elements[0].bounds.yMin < 30.0f);
    sigil_free_scene(scene);
}

TEST(test_stroke_bevel_join) {
    /* Same V-shape with bevel: no miter spike */
    const char *svg =
        "<svg viewBox=\"0 0 200 200\">"
        "  <path d=\"M 20 170 L 100 30 L 180 170\" fill=\"none\" stroke=\"green\" stroke-width=\"20\" stroke-linejoin=\"bevel\"/>"
        "</svg>";
    SigilScene *scene = sigil_parse_svg(svg, strlen(svg));
    ASSERT(scene != NULL);
    ASSERT(scene->element_count == 1);
    ASSERT(scene->elements[0].curve_count > 0);
    /* Bevel join should NOT extend far above the vertex */
    ASSERT(scene->elements[0].bounds.yMin >= 18.0f); /* at most half-width above vertex */
    sigil_free_scene(scene);
}

TEST(test_stroke_round_join) {
    const char *svg =
        "<svg viewBox=\"0 0 200 200\">"
        "  <path d=\"M 20 170 L 100 30 L 180 170\" fill=\"none\" stroke=\"green\" stroke-width=\"20\" stroke-linejoin=\"round\"/>"
        "</svg>";
    SigilScene *scene = sigil_parse_svg(svg, strlen(svg));
    ASSERT(scene != NULL);
    ASSERT(scene->element_count == 1);
    ASSERT(scene->elements[0].curve_count > 0);
    /* Round join bounded by half-width circle around vertex */
    ASSERT(scene->elements[0].bounds.yMin >= 18.0f);
    sigil_free_scene(scene);
}

TEST(test_stroke_square_cap) {
    const char *svg =
        "<svg viewBox=\"0 0 200 200\">"
        "  <path d=\"M 50 100 L 150 100\" fill=\"none\" stroke=\"red\" stroke-width=\"20\" stroke-linecap=\"square\"/>"
        "</svg>";
    SigilScene *scene = sigil_parse_svg(svg, strlen(svg));
    ASSERT(scene != NULL);
    ASSERT(scene->element_count == 1);
    ASSERT(scene->elements[0].curve_count > 0);
    /* Square cap extends 10px (half-width) past endpoints */
    ASSERT(scene->elements[0].bounds.xMin < 50.0f);
    ASSERT(scene->elements[0].bounds.xMax > 150.0f);
    sigil_free_scene(scene);
}

TEST(test_stroke_round_cap) {
    const char *svg =
        "<svg viewBox=\"0 0 200 200\">"
        "  <path d=\"M 50 100 L 150 100\" fill=\"none\" stroke=\"red\" stroke-width=\"20\" stroke-linecap=\"round\"/>"
        "</svg>";
    SigilScene *scene = sigil_parse_svg(svg, strlen(svg));
    ASSERT(scene != NULL);
    ASSERT(scene->element_count == 1);
    ASSERT(scene->elements[0].curve_count > 0);
    /* Round cap extends half-width past endpoints */
    ASSERT(scene->elements[0].bounds.xMin < 50.0f);
    ASSERT(scene->elements[0].bounds.xMax > 150.0f);
    sigil_free_scene(scene);
}

TEST(test_stroke_group_inherit_linejoin) {
    const char *svg =
        "<svg viewBox=\"0 0 200 200\">"
        "  <g stroke-linejoin=\"bevel\">"
        "    <path d=\"M 20 170 L 100 30 L 180 170\" fill=\"none\" stroke=\"green\" stroke-width=\"20\"/>"
        "  </g>"
        "</svg>";
    SigilScene *scene = sigil_parse_svg(svg, strlen(svg));
    ASSERT(scene != NULL);
    ASSERT(scene->element_count == 1);
    /* Should be bevel (inherited), so no miter spike */
    ASSERT(scene->elements[0].bounds.yMin >= 18.0f);
    sigil_free_scene(scene);
}

TEST(test_stroke_closed_path) {
    /* Closed triangle: all vertices should have joins */
    const char *svg =
        "<svg viewBox=\"0 0 200 200\">"
        "  <path d=\"M 50 150 L 100 50 L 150 150 Z\" fill=\"none\" stroke=\"blue\" stroke-width=\"10\"/>"
        "</svg>";
    SigilScene *scene = sigil_parse_svg(svg, strlen(svg));
    ASSERT(scene != NULL);
    ASSERT(scene->element_count == 1);
    ASSERT(scene->elements[0].curve_count > 0);
    sigil_free_scene(scene);
}

TEST(test_stroke_miter_limit) {
    /* Very acute angle with low miter limit should fall back to bevel */
    const char *svg =
        "<svg viewBox=\"0 0 400 200\">"
        "  <path d=\"M 20 100 L 200 10 L 20 20\" fill=\"none\" stroke=\"green\" stroke-width=\"10\" stroke-miterlimit=\"1\"/>"
        "</svg>";
    SigilScene *scene = sigil_parse_svg(svg, strlen(svg));
    ASSERT(scene != NULL);
    ASSERT(scene->element_count == 1);
    ASSERT(scene->elements[0].curve_count > 0);
    sigil_free_scene(scene);
}

/* ================================================================== */
/*  Integration tests                                                 */
/* ================================================================== */

TEST(test_integration_multi_element) {
    const char *svg =
        "<svg viewBox=\"0 0 400 400\">"
        "  <rect x=\"0\" y=\"0\" width=\"400\" height=\"400\" fill=\"#1a1a2e\"/>"
        "  <g transform=\"translate(200,200)\">"
        "    <circle cx=\"0\" cy=\"0\" r=\"150\" fill=\"#16213e\"/>"
        "    <ellipse cx=\"0\" cy=\"20\" rx=\"100\" ry=\"80\" fill=\"#0f3460\"/>"
        "    <path d=\"M -80 -40 Q 0 -120 80 -40\" fill=\"none\" stroke=\"#e94560\" stroke-width=\"4\"/>"
        "    <polygon points=\"-30,-60 0,-90 30,-60\" fill=\"#e94560\"/>"
        "  </g>"
        "  <rect x=\"10\" y=\"10\" width=\"80\" height=\"30\" fill=\"#e94560\" opacity=\"0.7\"/>"
        "</svg>";
    SigilScene *scene = sigil_parse_svg(svg, strlen(svg));
    ASSERT(scene != NULL);
    ASSERT(scene->element_count >= 5);
    /* Verify transforms were applied (circle at 200,200 with r=150) */
    SigilElement *circle = &scene->elements[1];
    ASSERT(circle->bounds.xMin < 100);
    ASSERT(circle->bounds.xMax > 300);
    /* Verify opacity on last rect */
    SigilElement *lastRect = &scene->elements[scene->element_count - 1];
    ASSERT_NEAR(lastRect->opacity, 0.7f, 0.01f);
    sigil_free_scene(scene);
}

TEST(test_parse_svg_text) {
    const char *svg = "<svg><text x=\"10\" y=\"50\" font-size=\"24\" fill=\"white\">Hello</text></svg>";
    SigilScene *scene = sigil_parse_svg(svg, strlen(svg));
    ASSERT(scene != NULL);
    /* Without a loaded font, text produces 0 elements -- that's OK */
    sigil_free_scene(scene);
}

/* ================================================================== */
/*  Main                                                              */
/* ================================================================== */

int main(void) {
    printf("SigilVG parser tests:\n");

    printf("\n-- Path parser --\n");
    RUN(test_parse_moveto_lineto);
    RUN(test_parse_relative_lineto);
    RUN(test_parse_hv_lines);
    RUN(test_parse_cubic);
    RUN(test_parse_quadratic);
    RUN(test_parse_smooth_cubic);
    RUN(test_parse_arc);
    RUN(test_parse_comma_separated);

    printf("\n-- SVG parsing --\n");
    RUN(test_parse_svg_rect);
    RUN(test_parse_svg_circle);
    RUN(test_parse_svg_path);
    RUN(test_parse_svg_group_transform);
    RUN(test_parse_svg_multiple_elements);
    RUN(test_parse_svg_ellipse);
    RUN(test_parse_svg_polyline);
    RUN(test_parse_color_names);
    RUN(test_parse_fill_rule);
    RUN(test_stroke_generates_fill);

    printf("\n-- Stroke join/cap tests --\n");
    RUN(test_stroke_miter_join);
    RUN(test_stroke_bevel_join);
    RUN(test_stroke_round_join);
    RUN(test_stroke_square_cap);
    RUN(test_stroke_round_cap);
    RUN(test_stroke_group_inherit_linejoin);
    RUN(test_stroke_closed_path);
    RUN(test_stroke_miter_limit);

    printf("\n-- Integration tests --\n");
    RUN(test_integration_multi_element);
    RUN(test_parse_svg_text);

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
