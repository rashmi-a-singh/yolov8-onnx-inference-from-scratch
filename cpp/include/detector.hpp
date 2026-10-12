// C5 — detector.hpp
#pragma once
#include <onnxruntime_cxx_api.h>
#include <cstdint>
#include <string>
#include <vector>

struct TensorInfo {
    std::string name;
    std::vector<int64_t> shape;          // -1 = dynamic dim
    ONNXTensorElementDataType type;
};

// RAII owner of the ORT environment + session.
// Ort:: types are move-only, so Detector is implicitly non-copyable.
class Detector {
public:
    Detector(const std::string& model_path, int intra_threads);

    const std::vector<TensorInfo>& inputs()  const { return inputs_; }
    const std::vector<TensorInfo>& outputs() const { return outputs_; }
    int intra_threads() const { return intra_threads_; }

    // One forward pass. `input` must hold exactly prod(input shape) floats.
    // ORT wraps this buffer WITHOUT copying, so it must stay alive until Run() returns.
    // The returned Ort::Value owns the output memory (allocated by ORT).
    std::vector<Ort::Value> run(std::vector<float>& input);

private:
    int intra_threads_;
    Ort::Env env_;               // must be declared BEFORE session_:
    Ort::SessionOptions opts_;   // members are initialised in declaration order,
    Ort::Session session_;       // and the session needs a live env + options
    Ort::MemoryInfo mem_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<TensorInfo> inputs_, outputs_;
};

std::string shape_str(const std::vector<int64_t>& s);
const char* type_str(ONNXTensorElementDataType t);
