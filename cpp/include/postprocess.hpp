// C16 — postprocess.hpp
#pragma once
#include "preprocess.hpp"
#include <vector>

struct Det {
    float x1, y1, x2, y2;   // corners
    float score;            // max class score (YOLOv8 has no objectness)
    int   cls;              // argmax class, 0..79
    int   anchor;           // column in the 84x8400 output (used for cross-checks)
};

// (1, 4+C, N) channel-major output -> candidates with score > conf, in anchor order.
// Indexes the tensor in place: no transpose copy.
std::vector<Det> decode(const float* out, int num_ch, int num_anchors, float conf);

// IoU with guards: no overlap -> 0, zero/negative union -> 0.
double iou(const Det& a, const Det& b);

// Greedy per-class NMS, written from scratch. Output ordered by (class asc, score desc).
std::vector<Det> nms(const std::vector<Det>& dets, float iou_thr);

// Reference only: same per-class grouping, each class through cv::dnn::NMSBoxes (as NB1 does).
std::vector<Det> nms_opencv(const std::vector<Det>& dets, float iou_thr);

// Reverse letterbox exactly like NB1: float32 (x - pad) / ratio, clip to [0,W] / [0,H].
void scale_back(std::vector<Det>& dets, const LetterboxInfo& info, int orig_w, int orig_h);

extern const char* const COCO80[80];
