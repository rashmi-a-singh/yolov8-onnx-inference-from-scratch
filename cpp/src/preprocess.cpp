#include "preprocess.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <fstream>
#include <stdexcept>

cv::Mat letterbox(const cv::Mat& bgr, int size, LetterboxInfo& info, int pad_value) {
    CV_Assert(!bgr.empty() && bgr.type() == CV_8UC3);
    const int h = bgr.rows, w = bgr.cols;
    info.ratio    = static_cast<double>(size) / std::max(h, w);  // Python float == IEEE double
    info.new_w    = static_cast<int>(w * info.ratio);             // int() truncates toward 0
    info.new_h    = static_cast<int>(h * info.ratio);
    info.pad_left = (size - info.new_w) / 2;                      // == Python // for non-negative ints
    info.pad_top  = (size - info.new_h) / 2;
    info.dw       = (size - info.new_w) / 2.0;
    info.dh       = (size - info.new_h) / 2.0;

    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(info.new_w, info.new_h), 0, 0, cv::INTER_LINEAR);
    cv::Mat canvas(size, size, CV_8UC3, cv::Scalar::all(pad_value));
    resized.copyTo(canvas(cv::Rect(info.pad_left, info.pad_top, info.new_w, info.new_h)));
    return canvas;
}

void to_chw_float(const cv::Mat& img, std::vector<float>& out, Norm norm) {
    CV_Assert(img.type() == CV_8UC3);
    const int H = img.rows, W = img.cols;
    const size_t HW = static_cast<size_t>(H) * W;
    out.resize(3 * HW);
    float* r = out.data();          // plane 0 = R
    float* g = r + HW;              // plane 1 = G
    float* b = g + HW;              // plane 2 = B
    const float inv = static_cast<float>(1.0 / 255.0);

    for (int y = 0; y < H; ++y) {
        const uint8_t* row = img.ptr<uint8_t>(y);   // per-row pointer: correct even if !isContinuous()
        for (int x = 0; x < W; ++x) {
            const uint8_t* px = row + 3 * x;         // HWC interleaved: B, G, R
            const size_t i = static_cast<size_t>(y) * W + x;
            if (norm == Norm::Divide) {
                r[i] = px[2] / 255.0f;  g[i] = px[1] / 255.0f;  b[i] = px[0] / 255.0f;
            } else {
                r[i] = px[2] * inv;     g[i] = px[1] * inv;     b[i] = px[0] * inv;
            }
        }
    }
}

void write_bin(const std::string& path, const void* data, size_t bytes) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    f.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
}
