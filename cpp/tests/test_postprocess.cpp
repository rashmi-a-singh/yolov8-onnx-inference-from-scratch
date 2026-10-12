// C22 — unit tests for decode / iou / nms (no framework; CHECK always runs, unlike assert under NDEBUG)
#include "postprocess.hpp"
#include <cmath>
#include <cstdio>
#include <vector>

static int g_pass = 0, g_fail = 0;
#define CHECK(cond) do { if (cond) ++g_pass; else { ++g_fail; \
    std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); } } while (0)

static Det box(float x1, float y1, float x2, float y2, float s, int c, int anchor) {
    return {x1, y1, x2, y2, s, c, anchor};
}

static void test_identical() {
    Det a = box(0, 0, 10, 10, 0.9f, 0, 0), b = box(0, 0, 10, 10, 0.8f, 0, 1);
    CHECK(iou(a, b) == 1.0);
    auto k = nms({a, b}, 0.45f);
    CHECK(k.size() == 1 && k[0].anchor == 0);              // higher score survives
}

static void test_disjoint_and_touching() {
    Det a = box(0, 0, 10, 10, 0.9f, 0, 0), b = box(20, 20, 30, 30, 0.8f, 0, 1);
    Det t = box(10, 0, 20, 10, 0.7f, 0, 2);               // shares an edge with a
    CHECK(iou(a, b) == 0.0);
    CHECK(iou(a, t) == 0.0);
    CHECK(nms({a, b, t}, 0.45f).size() == 3);
}

static void test_iou_at_threshold() {
    // inter = 50, union = 100 -> IoU exactly 0.5 (exact in float and double; 0.45 would not be)
    Det a = box(0, 0, 10, 10, 0.9f, 0, 0), b = box(0, 0, 10, 5, 0.8f, 0, 1);
    CHECK(iou(a, b) == 0.5);
    CHECK(nms({a, b}, 0.50f).size() == 2);                 // suppress only if IoU > thr
    CHECK(nms({a, b}, 0.49f).size() == 1);
}

static void test_zero_area() {
    Det z  = box(5, 5, 5, 5, 0.9f, 0, 0);                  // degenerate point box
    Det a  = box(0, 0, 10, 10, 0.8f, 0, 1);
    Det z2 = box(5, 5, 5, 5, 0.7f, 0, 2);
    CHECK(iou(z, a) == 0.0 && !std::isnan(iou(z, a)));
    CHECK(iou(z, z2) == 0.0 && !std::isnan(iou(z, z2)));  // union 0 -> guarded, no 0/0 NaN
    CHECK(nms({z, a, z2}, 0.45f).size() == 3);
}

static void test_two_classes_overlap() {
    Det b = box(0, 0, 10, 10, 0.8f, 1, 1), a = box(0, 0, 10, 10, 0.9f, 0, 0);
    auto k = nms({b, a}, 0.45f);
    CHECK(k.size() == 2);                                   // different classes never suppress
    CHECK(k.size() == 2 && k[0].cls == 0 && k[1].cls == 1); // output ordered by class
}

static void test_empty() {
    CHECK(nms({}, 0.45f).empty());
}

static void test_decode() {
    const int N = 3, CH = 84;
    std::vector<float> out(CH * N, 0.0f);
    auto at = [&](int c, int a) -> float& { return out[c * N + a]; };   // channel-major (84, N)
    at(0, 0) = 50; at(1, 0) = 60; at(2, 0) = 20; at(3, 0) = 10; at(4 + 7, 0) = 0.6f;  // anchor 0 -> class 7
    at(4 + 2, 1) = 0.7f; at(4 + 9, 1) = 0.7f;   // anchor 1: tie -> first max (class 2), like np.argmax
    at(4 + 0, 2) = 0.25f;                        // anchor 2: exactly at conf -> rejected by strict >
    auto d = decode(out.data(), CH, N, 0.25f);
    CHECK(d.size() == 2);
    if (d.size() == 2) {
        CHECK(d[0].anchor == 0 && d[0].cls == 7);
        CHECK(d[0].x1 == 40 && d[0].y1 == 55 && d[0].x2 == 60 && d[0].y2 == 65);
        CHECK(d[1].anchor == 1 && d[1].cls == 2);
    }
}

int main() {
    test_identical();
    test_disjoint_and_touching();
    test_iou_at_threshold();
    test_zero_area();
    test_two_classes_overlap();
    test_empty();
    test_decode();
    std::printf("TESTS passed=%d failed=%d\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
