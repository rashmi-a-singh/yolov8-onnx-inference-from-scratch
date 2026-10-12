// C28 — unit tests for the COCO helpers
#include "coco.hpp"
#include <cstdio>
#include <sstream>
#include <stdexcept>

static int g_pass = 0, g_fail = 0;
#define CHECK(cond) do { if (cond) ++g_pass; else { ++g_fail; \
    std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); } } while (0)

int main() {
    CHECK(coco80_to_coco91(0)  == 1);    // person
    CHECK(coco80_to_coco91(5)  == 6);    // bus
    CHECK(coco80_to_coco91(11) == 13);   // stop sign: first gap (COCO id 12 unused)
    CHECK(coco80_to_coco91(79) == 90);   // toothbrush
    bool threw = false;
    try { coco80_to_coco91(80); } catch (const std::out_of_range&) { threw = true; }
    CHECK(threw);

    CHECK(coco_image_id("/x/val2017/000000397133.jpg") == 397133);
    CHECK(coco_image_id("bus.jpg") == -1);

    std::ostringstream os; bool first = true;
    append_coco_json(os, 42, {Det{10, 20, 40, 60, 0.5f, 5, 0}}, first);   // xyxy -> xywh, bus -> 6
    CHECK(os.str() == "\n{\"image_id\":42,\"category_id\":6,\"bbox\":[10,20,30,40],\"score\":0.5}");

    std::printf("TESTS passed=%d failed=%d\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
