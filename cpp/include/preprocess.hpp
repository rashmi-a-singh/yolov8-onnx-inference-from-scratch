#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>

struct LetterboxInfo {
    double ratio = 1.0;              // size / max(h, w)
    int new_w = 0, new_h = 0;
    int pad_left = 0, pad_top = 0;   // integer offsets actually used for the paste
    double dw = 0.0, dh = 0.0;       // float padding exactly as NB1 returns it (used for scale-back)
};

// Exact port of NB1 letterbox(): int() truncation, //2 centring, pad 114, INTER_LINEAR.
cv::Mat letterbox(const cv::Mat& bgr, int size, LetterboxInfo& info, int pad_value = 114);

// Divide        == NumPy `astype(float32) / 255.0`   (pipeline default)
// MulReciprocal == x * float(1/255)                  (layout check vs blobFromImage)
enum class Norm { Divide, MulReciprocal };

// Hand-written BGR->RGB, /255, HWC->CHW into a contiguous buffer of 3*H*W floats.
void to_chw_float(const cv::Mat& bgr_u8, std::vector<float>& out, Norm norm = Norm::Divide);

void write_bin(const std::string& path, const void* data, size_t bytes);
