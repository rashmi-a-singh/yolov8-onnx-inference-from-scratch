// C30 — main.cpp v4: batch CLI + COCO JSON + per-stage timing + --bench
#include "coco.hpp"
#include "detector.hpp"
#include "postprocess.hpp"
#include "preprocess.hpp"
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using Args  = std::map<std::string, std::string>;

static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k.rfind("--", 0) != 0) throw std::runtime_error("bad arg: " + k);
        bool has_val = i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0;
        a[k.substr(2)] = has_val ? argv[++i] : "1";
    }
    return a;
}

struct StageMs {
    double pre = 0, infer = 0, post = 0, scale = 0;
    double total() const { return pre + infer + post + scale; }
};

// One image through the full pipeline. `buf` is reused across calls: no per-frame reallocation.
// Timing excludes JPEG decode / disk I/O (a camera pipeline hands us decoded frames).
static std::vector<Det> run_pipeline(Detector& det, const cv::Mat& img, float conf, float iou_thr,
                                     int max_det, std::vector<float>& buf, StageMs& t, size_t& n_cands) {
    const int S = static_cast<int>(det.inputs()[0].shape[3]);

    auto t0 = Clock::now();
    LetterboxInfo info;
    cv::Mat lb = letterbox(img, S, info);
    to_chw_float(lb, buf, Norm::Divide);
    t.pre = ms_since(t0);

    t0 = Clock::now();
    auto out = det.run(buf);
    t.infer = ms_since(t0);

    t0 = Clock::now();
    const auto shape = out[0].GetTensorTypeAndShapeInfo().GetShape();
    auto cands = decode(out[0].GetTensorData<float>(), static_cast<int>(shape[1]),
                        static_cast<int>(shape[2]), conf);
    n_cands = cands.size();
    auto kept = nms(cands, iou_thr);
    if (max_det > 0 && static_cast<int>(kept.size()) > max_det) {
        std::stable_sort(kept.begin(), kept.end(),
                         [](const Det& x, const Det& y) { return x.score > y.score; });
        kept.resize(max_det);
    }
    t.post = ms_since(t0);

    t0 = Clock::now();
    scale_back(kept, info, img.cols, img.rows);
    t.scale = ms_since(t0);
    return kept;
}

static std::vector<std::string> list_images(const Args& a) {
    std::vector<std::string> v;
    if (a.count("image")) v.push_back(a.at("image"));
    if (a.count("images")) {
        const fs::path dir = a.at("images");
        if (a.count("list")) {
            std::ifstream f(a.at("list"));
            if (!f) throw std::runtime_error("cannot open list " + a.at("list"));
            for (std::string line; std::getline(f, line);) {
                while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
                if (!line.empty()) v.push_back((dir / line).string());
            }
        } else {
            for (const auto& e : fs::directory_iterator(dir)) {
                std::string ext = e.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(),
                               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") v.push_back(e.path().string());
            }
            std::sort(v.begin(), v.end());
        }
    }
    if (a.count("limit")) v.resize(std::min(v.size(), static_cast<size_t>(std::stoul(a.at("limit")))));
    return v;
}

// population std (ddof=0) to match np.std in NB1
static std::pair<double, double> mean_std(const std::vector<double>& x) {
    double m = 0; for (double v : x) m += v; m /= x.size();
    double s = 0; for (double v : x) s += (v - m) * (v - m);
    return {m, std::sqrt(s / x.size())};
}

static void print_stage(const char* name, const std::vector<double>& x) {
    auto [m, s] = mean_std(x);
    std::cout << std::fixed << std::setprecision(4)
              << "BENCH_STAGE " << name << " mean_ms=" << m << " std_ms=" << s << "\n";
}

int main(int argc, char** argv) {
    try {
        Args a = parse_args(argc, argv);

        if (a.count("dump-coco-map")) {
            for (int i = 0; i < 80; ++i)
                std::cout << "MAP " << i << " " << coco80_to_coco91(i) << " " << COCO80[i] << "\n";
            return 0;
        }
        if (!a.count("model") || !(a.count("image") || a.count("images"))) {
            std::cerr << "usage: detector --model X.onnx (--image F | --images DIR [--list names.txt]) "
                         "[--out dets.json] [--conf 0.25] [--iou 0.45] [--max-det 0] [--threads 1] "
                         "[--limit N] [--print] [--bench [--warmup 3] [--iters 50]]\n";
            return 2;
        }
        const int   threads = a.count("threads") ? std::stoi(a["threads"]) : 1;
        const float conf    = a.count("conf")    ? std::stof(a["conf"])    : 0.25f;
        const float iou_thr = a.count("iou")     ? std::stof(a["iou"])     : 0.45f;
        const int   max_det = a.count("max-det") ? std::stoi(a["max-det"]) : 0;   // 0 = unlimited (NB1 demo)

        std::cout << "ORT_CPP_VERSION " << OrtGetApiBase()->GetVersionString() << "\n";
        Detector det(a["model"], threads);
        std::cout << "THREADS intra=" << det.intra_threads() << "\n";
        const auto files = list_images(a);
        if (files.empty()) throw std::runtime_error("no images found");
        std::vector<float> buf;   // reused input buffer

        // ── benchmark mode: one image, warm-up + timed iterations ──
        if (a.count("bench")) {
            const int warm  = a.count("warmup") ? std::stoi(a["warmup"]) : 3;
            const int iters = a.count("iters")  ? std::stoi(a["iters"])  : 50;
            cv::Mat img = cv::imread(files[0], cv::IMREAD_COLOR);
            if (img.empty()) throw std::runtime_error("cannot read " + files[0]);

            // (a) model-only: Run() on a fixed preprocessed tensor, same protocol as NB1's .run() benchmark
            LetterboxInfo info;
            to_chw_float(letterbox(img, static_cast<int>(det.inputs()[0].shape[3]), info), buf, Norm::Divide);
            std::vector<double> model_only;
            for (int i = 0; i < warm + iters; ++i) {
                auto t0 = Clock::now();
                auto out = det.run(buf);
                double ms = ms_since(t0);
                if (i >= warm) model_only.push_back(ms);
            }

            // (b) end-to-end per image, with per-stage breakdown
            std::vector<double> pre, inf, post, scale, total;
            for (int i = 0; i < warm + iters; ++i) {
                StageMs t; size_t nc = 0;
                run_pipeline(det, img, conf, iou_thr, max_det, buf, t, nc);
                if (i >= warm) {
                    pre.push_back(t.pre); inf.push_back(t.infer); post.push_back(t.post);
                    scale.push_back(t.scale); total.push_back(t.total());
                }
            }
            std::cout << "BENCH image=" << files[0] << " threads=" << threads
                      << " warmup=" << warm << " iters=" << iters << "\n";
            print_stage("model_only", model_only);
            print_stage("e2e_preprocess", pre);
            print_stage("e2e_inference", inf);
            print_stage("e2e_decode_nms", post);
            print_stage("e2e_scale_back", scale);
            print_stage("e2e_total", total);
            return 0;
        }

        // ── batch mode: every image -> COCO results JSON ──
        std::ostringstream json;
        json << "[";
        bool first = true;
        size_t n_ok = 0, n_skip = 0, n_cands_total = 0, n_dets_total = 0;
        StageMs sum;
        for (const auto& f : files) {
            cv::Mat img = cv::imread(f, cv::IMREAD_COLOR);
            if (img.empty()) { std::cerr << "WARN cannot read " << f << "\n"; ++n_skip; continue; }
            StageMs t; size_t nc = 0;
            auto dets = run_pipeline(det, img, conf, iou_thr, max_det, buf, t, nc);
            sum.pre += t.pre; sum.infer += t.infer; sum.post += t.post; sum.scale += t.scale;
            n_cands_total += nc; n_dets_total += dets.size(); ++n_ok;
            append_coco_json(json, coco_image_id(f), dets, first);
            if (a.count("print"))
                for (const auto& d : dets)
                    std::cout << std::setprecision(9) << "DET " << fs::path(f).filename().string() << " "
                              << d.cls << " " << d.score << " " << d.x1 << " " << d.y1 << " "
                              << d.x2 << " " << d.y2 << "\n";
        }
        json << "\n]\n";
        if (a.count("out")) {
            std::ofstream o(a["out"]);
            if (!o) throw std::runtime_error("cannot write " + a["out"]);
            o << json.str();
        }
        std::cout << "PROCESSED images=" << n_ok << " skipped=" << n_skip
                  << " candidates=" << n_cands_total << " dets=" << n_dets_total << "\n"
                  << std::fixed << std::setprecision(2)
                  << "STAGE_SUM_MS pre=" << sum.pre << " infer=" << sum.infer
                  << " post=" << sum.post << " scale=" << sum.scale << "\n";
    } catch (const Ort::Exception& e) {
        std::cerr << "ORT error: " << e.what() << "\n"; return 1;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n"; return 1;
    }
    return 0;
}
