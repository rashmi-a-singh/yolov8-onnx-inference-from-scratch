# YOLOv8 Edge Inference: Python → C++17 → INT8 → Snapdragon NPU

YOLOv8 inference built by hand on ONNX Runtime (no Ultralytics at inference time), ported to C++17 with
bit-level parity checks, quantized to INT8, and profiled on a Snapdragon 8 Gen 3 NPU via Qualcomm AI Hub.

Every number below was produced by a committed (top-to-bottom) notebook run and is recorded in
`results.json` with the cell that produced it. Each notebook ends with a **claim ledger** mapping every
claim to its evidence (`results/claim_ledger_nb2.json`, `results/claim_ledger_nb3.json`).

## Results at a glance

| | Result |
|---|---|
| C++ ↔ Python parity | **Bit-identical**: preprocessed tensor and raw `output0` max diff 0.0; **550/550** detections matched on 100 COCO images |
| C++ quality gates | **26/26** unit tests; **ASan + UBSan + LeakSanitizer clean**; scratch NMS keeps exactly the same boxes as `cv::dnn::NMSBoxes` on 20/20 COCO images at demo and eval thresholds |
| C++ vs Python speed (same ORT core) | model-only within 3.2%; pre-processing **2.7–3.1×** and decode+NMS **2.1–3.3×** faster in C++; end-to-end only 1.02–1.08× because inference dominates |
| Full INT8 (all nodes) | **0 detections**: YOLOv8's output concatenates boxes (0–640) and scores (0–1) under one uint8 scale of 2.54, so every score rounds to 0 |
| Best INT8 (decode tail in FP32 + reduce_range) | **mAP@0.5:0.95 0.381 → 0.366** (−3.9%), **12.24 → 3.49 MB** (3.5× smaller), 500-image COCO val2017 subset |
| Snapdragon 8 Gen 3 (Galaxy S24), Qualcomm AI Hub | **3.34 → 2.19 ms** FP32 → INT8 (1.52×), **251/251 layers on the NPU**, no CPU fallback |
| On-device parity | FP32 on device vs desktop: **84/84 detections matched** (10 images) |

## Notebooks

| # | Notebook | What it proves |
|---|---|---|
| 1 | `notebooks/01_yolov8_onnx_inference_python.ipynb` | Manual Python pipeline: letterbox, normalisation, HWC→CHW, decode, per-class NMS, inverse letterbox; n/s/m CPU benchmark |
| 2 | `notebooks/02_yolov8_onnx_inference_cpp.ipynb` | C++17 port on the ONNX Runtime C++ API; stage-by-stage parity with Python; unit tests; sanitizers; 100-image parity; C++ vs Python benchmarks |
| 3 | `notebooks/03_yolov8_int8_quantization_snapdragon.ipynb` | COCO mAP harness; INT8 PTQ ablation; diagnosis of the full-INT8 collapse; Qualcomm AI Hub compile, profile and on-device inference |

## 2 · C++17 port

- `Detector`: RAII owner of `Ort::Env` / `Ort::Session`; I/O names and shapes read from the model.
- Input wrapped with `Ort::Value::CreateTensor` as a **no-copy view** over a reused `std::vector<float>`.
- Hand-written BGR→RGB, /255, HWC→CHW loop; **divides** by 255 to match NumPy exactly (multiplying by 1/255 differs by 1 ulp on 57% of elements).
- `decode()` indexes the channel-major `(1, 84, 8400)` output in place, with channels in the outer loop so the inner loop has stride 1 (no transpose copy).
- Greedy **per-class NMS written from scratch**, with tests for identical, disjoint, touching and zero-area boxes, IoU exactly at threshold, overlapping classes, and empty input.
- Batch CLI writing COCO-format JSON (YOLO-80 → COCO-91 mapping validated by name against `instances_val2017.json`).
- Performance finding: at 4 threads, ONNX Runtime's intra-op workers spin-wait after `Run()` and slow the single-threaded pre/post-processing; `session.intra_op.allow_spinning=0` cut inference jitter (std) from 12.8 to 3.5 ms.

```bash
# needs OpenCV (C++), CMake ≥ 3.16 and the prebuilt onnxruntime-linux-x64 package
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release -DORT_DIR=/path/to/onnxruntime-linux-x64-1.27.0
cmake --build build -j
./build/test_postprocess && ./build/test_coco
./build/detector --model yolov8n.onnx --images val2017/ --list ids.txt --out dets.json --conf 0.25 --iou 0.45 --threads 4
./build/detector --model yolov8n.onnx --image bus.jpg --bench          # 3 warm-up + 50 timed runs
# memory-safety build
cmake -S cpp -B build-asan -DCMAKE_BUILD_TYPE=Debug -DSANITIZE=ON -DORT_DIR=...
```

## 3 · INT8 quantization ablation (yolov8n)

QDQ format, per-channel symmetric int8 weights, asymmetric uint8 activations. Calibration on 300 COCO val
images, disjoint from the 500 eval images (overlap asserted 0). Eval: conf 0.001, NMS IoU 0.7, max_det 300.

| Config | What stays FP32 | mAP@0.5:0.95 | mAP@0.5 | Size |
|---|---|---|---|---|
| A · FP32 | everything | 0.3812 | 0.5138 | 12.24 MB |
| B · INT8 MinMax | nothing | 0.0000 (0 detections) | 0.0000 | 3.43 MB |
| E1 · INT8 MinMax | final Concat | 0.2835 | 0.4368 | 3.43 MB |
| E2 · INT8 MinMax | whole decode tail (DFL, box decode, Sigmoid, Concats) | 0.3368 | 0.4625 | 3.49 MB |
| C · INT8 Entropy | decode tail | 0.3421 | 0.4683 | 3.49 MB |
| D · INT8 Percentile 99.999 | decode tail | 0.3428 | 0.4723 | 3.49 MB |
| **E2_RR · INT8 MinMax + reduce_range** | decode tail | **0.3663** | **0.4999** | 3.49 MB |

What the table shows:
- **E1 loses much more on mAP@0.5:0.95 (−25.6%) than on mAP@0.5 (−15.0%).** Its box branch is still quantized at 2.57 px per step, which hurts localization at strict IoU thresholds.
- **reduce_range recovers about two-thirds of the remaining loss on this CPU** (AVX-512, no VNNI). That's consistent with U8×S8 saturation in `vpmaddubsw`. On the NPU there was no sign it helps (E2 vs E2_RR on-device parity 90% vs 88%).
- **INT8 on the Kaggle CPU:** 1.16× faster at 1 thread, no faster at 4 threads. **On the Snapdragon NPU:** 1.52×.

### Engineering problems solved along the way
- **Calibration would have needed 69 GB of RAM** (ORT keeps every activation of every image). I chunked `collect_data` and verified MinMax chunked is bit-identical to single-pass; Entropy chunked was validated by mAP (Δ +0.002).
- **The per-channel QDQ model failed to load** (`axis` attribute) because the export was opset 12. Converted to opset 13, with outputs verified identical.
- **AI Hub rejected the model** because the graph output also appeared in `value_info`. Stripped it and unused opset domains, with outputs verified bit-identical.

## Snapdragon 8 Gen 3 (Samsung Galaxy S24) via Qualcomm AI Hub — `qnn_dlc` target

| Model | Inference | Layers on NPU | CPU fallback | On-device vs desktop FP32 (10 images) |
|---|---|---|---|---|
| FP32 | 3.34 ms | 247/247 | none | 100% detections matched |
| INT8 E2 | 2.19 ms | 251/251 | none | 90% |
| INT8 E2_RR | 2.20 ms | 251/251 | none | 88% |

## Scope and limitations

- **Accuracy is measured on a 500-image COCO val2017 subset**, not full val. FP32 n/s/m score 0.381/0.463/0.512, within about 1.5 points of Ultralytics' full-val numbers.
- **Post-training quantization only**, no QAT.
- **On-device accuracy:** only 10-image detection parity was measured on the device, not mAP. INT8 on the device matches INT8 in ORT at 86%, because the two runtimes implement quantized arithmetic differently.
- **The Kaggle CPU is not an edge device.** CPU and NPU latencies are different hardware classes; compare FP32 vs INT8 within each, not across.
- **Single device and model:** one device (Galaxy S24), batch size 1, static 640×640 input, yolov8n for the INT8 study.

## Repository layout

```
notebooks/   three Kaggle notebooks, committed runs with outputs
cpp/         CMakeLists.txt, include/, src/, tests/
results/     claim ledgers (resume claim → evidence → notebook cell)
```

## Background

At LSE I designed the ground-truth study and audited a production street-imagery pipeline
(1M+ images, YOLO26x detection → EfficientNet-B0 rider gate → EfficientNet-B3 + YOLO11m-cls ensemble).
This repository is a separate personal project focused on edge deployment.

*Stack: C++17 · Python · ONNX Runtime · OpenCV · NumPy · pycocotools · Qualcomm AI Hub · Ultralytics (export only)*
