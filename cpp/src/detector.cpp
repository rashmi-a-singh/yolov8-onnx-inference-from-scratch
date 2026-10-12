// C6 — detector.cpp
#include "detector.hpp"
#include <sstream>
#include <stdexcept>

namespace {

Ort::SessionOptions make_opts(int intra_threads) {
    Ort::SessionOptions o;
    o.SetIntraOpNumThreads(intra_threads);   // threads inside one op (GEMM/conv)
    o.SetInterOpNumThreads(1);               // no parallelism across ops
    o.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    o.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    return o;
}

std::vector<TensorInfo> read_io(Ort::Session& s, bool is_input) {
    Ort::AllocatorWithDefaultOptions alloc;
    const size_t n = is_input ? s.GetInputCount() : s.GetOutputCount();
    std::vector<TensorInfo> v;
    for (size_t i = 0; i < n; ++i) {
        Ort::AllocatedStringPtr name = is_input ? s.GetInputNameAllocated(i, alloc)
                                                : s.GetOutputNameAllocated(i, alloc);
        Ort::TypeInfo ti = is_input ? s.GetInputTypeInfo(i) : s.GetOutputTypeInfo(i);
        auto tsi = ti.GetTensorTypeAndShapeInfo();
        v.push_back({name.get(), tsi.GetShape(), tsi.GetElementType()});
    }
    return v;
}

}  // namespace

Detector::Detector(const std::string& model_path, int intra_threads)
    : intra_threads_(intra_threads),
      env_(ORT_LOGGING_LEVEL_WARNING, "yolov8"),
      opts_(make_opts(intra_threads)),
      session_(env_, model_path.c_str(), opts_),
      inputs_(read_io(session_, true)),
      outputs_(read_io(session_, false)) {}

std::vector<Ort::Value> Detector::run(std::vector<float>& input) {
    const auto& in = inputs_[0];
    size_t expected = 1;
    for (int64_t d : in.shape) {
        if (d <= 0) throw std::runtime_error("dynamic input dim not supported: " + shape_str(in.shape));
        expected *= static_cast<size_t>(d);
    }
    if (input.size() != expected)
        throw std::runtime_error("input has " + std::to_string(input.size()) +
                                 " floats, model expects " + std::to_string(expected));

    // No-copy view: ORT reads directly from input.data(); we keep ownership.
    Ort::Value x = Ort::Value::CreateTensor<float>(mem_, input.data(), input.size(),
                                                   in.shape.data(), in.shape.size());
    const char* in_names[]  = {inputs_[0].name.c_str()};
    const char* out_names[] = {outputs_[0].name.c_str()};
    return session_.Run(Ort::RunOptions{nullptr}, in_names, &x, 1, out_names, 1);
}

std::string shape_str(const std::vector<int64_t>& s) {
    std::ostringstream os;
    for (size_t i = 0; i < s.size(); ++i) os << (i ? "x" : "") << s[i];
    return os.str();
}

const char* type_str(ONNXTensorElementDataType t) {
    switch (t) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return "float32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return "float16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   return "uint8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:    return "int8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   return "int64";
        default:                                    return "other";
    }
}
