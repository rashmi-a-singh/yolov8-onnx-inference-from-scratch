// C17 — postprocess.cpp
#include "postprocess.hpp"
#include <opencv2/dnn.hpp>
#include <algorithm>
#include <map>
#include <numeric>

const char* const COCO80[80] = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush"};

std::vector<Det> decode(const float* out, int num_ch, int N, float conf) {
    const int C = num_ch - 4;
    const size_t n = static_cast<size_t>(N);
    // Channel-major layout: element (ch, a) lives at out[ch * N + a].
    // Channels in the OUTER loop -> inner loop walks memory with stride 1.
    // (Per-anchor argmax would jump N floats = 33.6 KB between consecutive class reads.)
    std::vector<float> best(out + 4 * n, out + 5 * n);   // class-0 row as the starting max
    std::vector<int>   best_cls(n, 0);
    for (int c = 1; c < C; ++c) {
        const float* row = out + (4 + c) * n;
        for (int a = 0; a < N; ++a)
            if (row[a] > best[a]) { best[a] = row[a]; best_cls[a] = c; }  // strict > == np.argmax (first max wins)
    }
    const float* cx = out;
    const float* cy = out + n;
    const float* w  = out + 2 * n;
    const float* h  = out + 3 * n;

    std::vector<Det> dets;
    for (int a = 0; a < N; ++a) {
        if (!(best[a] > conf)) continue;                  // same strict > as NB1's mask
        const float hw = w[a] / 2.0f, hh = h[a] / 2.0f;   // float32 math, same as NumPy
        dets.push_back({cx[a] - hw, cy[a] - hh, cx[a] + hw, cy[a] + hh, best[a], best_cls[a], a});
    }
    return dets;
}

double iou(const Det& a, const Det& b) {
    const double iw = std::max(0.0, static_cast<double>(std::min(a.x2, b.x2)) - std::max(a.x1, b.x1));
    const double ih = std::max(0.0, static_cast<double>(std::min(a.y2, b.y2)) - std::max(a.y1, b.y1));
    const double inter  = iw * ih;
    const double area_a = std::max(0.0, static_cast<double>(a.x2) - a.x1) * std::max(0.0, static_cast<double>(a.y2) - a.y1);
    const double area_b = std::max(0.0, static_cast<double>(b.x2) - b.x1) * std::max(0.0, static_cast<double>(b.y2) - b.y1);
    const double uni = area_a + area_b - inter;
    return uni > 0.0 ? inter / uni : 0.0;
}

std::vector<Det> nms(const std::vector<Det>& dets, float iou_thr) {
    // Group by class, highest score first within a class. stable_sort keeps anchor order on ties.
    std::vector<int> order(dets.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int i, int j) {
        if (dets[i].cls != dets[j].cls) return dets[i].cls < dets[j].cls;
        return dets[i].score > dets[j].score;
    });

    std::vector<Det> keep;
    size_t cls_start = 0;   // where the current class's kept boxes begin inside `keep`
    for (int idx : order) {
        const Det& d = dets[idx];
        if (keep.empty() || keep.back().cls != d.cls) cls_start = keep.size();
        bool suppressed = false;
        for (size_t k = cls_start; k < keep.size(); ++k)    // compare only against same-class keepers
            if (iou(keep[k], d) > iou_thr) { suppressed = true; break; }
        if (!suppressed) keep.push_back(d);
    }
    return keep;
}

std::vector<Det> nms_opencv(const std::vector<Det>& dets, float iou_thr) {
    std::map<int, std::vector<int>> by_cls;
    for (int i = 0; i < static_cast<int>(dets.size()); ++i) by_cls[dets[i].cls].push_back(i);

    std::vector<Det> keep;
    for (const auto& kv : by_cls) {
        const std::vector<int>& idxs = kv.second;
        std::vector<cv::Rect2d> boxes;
        std::vector<float> scores;
        for (int i : idxs) {
            const Det& d = dets[i];
            const float w = d.x2 - d.x1, h = d.y2 - d.y1;   // float32, like NB1's xywh array
            boxes.emplace_back(d.x1, d.y1, w, h);
            scores.push_back(d.score);
        }
        std::vector<int> kept;
        cv::dnn::NMSBoxes(boxes, scores, 0.0f, iou_thr, kept);
        for (int k : kept) keep.push_back(dets[idxs[k]]);
    }
    return keep;
}

void scale_back(std::vector<Det>& dets, const LetterboxInfo& info, int orig_w, int orig_h) {
    const float pw = static_cast<float>(info.dw), ph = static_cast<float>(info.dh);
    const float r  = static_cast<float>(info.ratio);
    const float W  = static_cast<float>(orig_w),  H  = static_cast<float>(orig_h);
    for (auto& d : dets) {
        d.x1 = std::clamp((d.x1 - pw) / r, 0.0f, W);
        d.y1 = std::clamp((d.y1 - ph) / r, 0.0f, H);
        d.x2 = std::clamp((d.x2 - pw) / r, 0.0f, W);
        d.y2 = std::clamp((d.y2 - ph) / r, 0.0f, H);
    }
}
