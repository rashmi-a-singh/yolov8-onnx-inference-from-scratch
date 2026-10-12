// C27 — coco.cpp
#include "coco.hpp"
#include <filesystem>
#include <iomanip>
#include <stdexcept>

static const int kCoco91[80] = {
     1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90};

int coco80_to_coco91(int c) {
    if (c < 0 || c >= 80) throw std::out_of_range("class index " + std::to_string(c));
    return kCoco91[c];
}

long long coco_image_id(const std::string& path) {
    const std::string stem = std::filesystem::path(path).stem().string();
    if (stem.empty() || stem.find_first_not_of("0123456789") != std::string::npos) return -1;
    return std::stoll(stem);
}

void append_coco_json(std::ostream& os, long long image_id, const std::vector<Det>& dets, bool& first) {
    os << std::setprecision(9);   // 9 significant digits round-trip any float32 exactly
    for (const auto& d : dets) {
        os << (first ? "\n" : ",\n")
           << "{\"image_id\":" << image_id
           << ",\"category_id\":" << coco80_to_coco91(d.cls)
           << ",\"bbox\":[" << d.x1 << "," << d.y1 << "," << (d.x2 - d.x1) << "," << (d.y2 - d.y1) << "]"
           << ",\"score\":" << d.score << "}";
        first = false;
    }
}
