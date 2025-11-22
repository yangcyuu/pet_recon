#pragma once

#include "utils.h"

struct TextureEvalParams {
  torch::nn::functional::GridSampleFuncOptions::mode_t mode = torch::kBilinear;
  torch::nn::functional::GridSampleFuncOptions::padding_mode_t padding_mode = torch::kZeros;
  std::optional<bool> align_corners = std::nullopt;
};

class Texture2D {
public:
  Texture2D() = default;
  Texture2D(torch::Tensor data) : _data(std::move(data)) {
    if (_data.dim() != 3) {
      ERROR_AND_EXIT("Texture2D data must be a 3D tensor (C, H, W)");
    }
    if (!_data.is_contiguous()) {
      _data = _data.contiguous();
    }
  }

  Texture2D(int64_t width, int64_t height, int64_t channels, const torch::TensorOptions &options = {}) :
      _data(torch::empty({height, width, channels}, options)) {}

  Texture2D(void *data_ptr, int64_t width, int64_t height, int64_t channels, const torch::TensorOptions &options = {}) :
      _data(torch::from_blob(data_ptr, {height, width, channels}, options).clone()) {}

  template<typename T>
    requires(std::is_fundamental_v<T>)
  Texture2D(const T &value, int64_t width, int64_t height, int64_t channels, const torch::TensorOptions &options = {}) :
      _data(torch::full({height, width, channels}, value, options)) {}

  static Texture2D cuda(int64_t width, int64_t height, int64_t channels, const torch::TensorOptions &options = {}) {
    Texture2D tex;
    tex._data = torch::empty({height, width, channels}, options.device(torch::kCUDA));
    return tex;
  }

  static Texture2D cuda(void *data_ptr, int64_t width, int64_t height, int64_t channels,
                        const torch::TensorOptions &options = {}) {
    Texture2D tex;
    if (!options.has_device() || options.device().is_cpu()) {
      tex._data = torch::from_blob(data_ptr, {height, width, channels}, options).to(torch::kCUDA);
    } else if (options.device().is_cuda()) {
      tex._data = torch::from_blob(data_ptr, {height, width, channels}, options).clone();
    } else {
      ERROR_AND_EXIT("Unsupported device type for Texture2D::cuda");
    }
    return tex;
  }

  template<typename T>
    requires(std::is_fundamental_v<T>)
  static Texture2D cuda(const T &value, int64_t width, int64_t height, int64_t channels,
                        const torch::TensorOptions &options = {}) {
    Texture2D tex;
    tex._data = torch::full({height, width, channels}, value, options.device(torch::kCUDA));
    return tex;
  }


  operator torch::Tensor() const { return _data; }

  const torch::Tensor &tensor() const { return _data; }

  torch::Tensor &tensor() { return _data; }

  int64_t width() const { return _data.size(1); }

  int64_t height() const { return _data.size(0); }

  int64_t channels() const { return _data.size(2); }

  // uv: [N, 2] with values in [-1, 1]
  torch::Tensor eval(const torch::Tensor &uv, const TextureEvalParams &params = {}) const;

  void assign(const torch::Tensor &rows, const torch::Tensor &cols, const torch::Tensor &channels,
              const torch::Tensor &values) {
    // rows, cols, channels, values: [N]
    _data.index_put_({rows, cols, channels}, values);
  }

  void save_rawdata(const std::string &filename) const;

private:
  torch::Tensor _data;
};

class Texture3D {
public:
  Texture3D() = default;
  Texture3D(torch::Tensor data) : _data(std::move(data)) {
    if (_data.dim() != 4) {
      ERROR_AND_EXIT("Texture3D data must be a 4D tensor (C, D, H, W)");
    }
    if (!_data.is_contiguous()) {
      _data = _data.contiguous();
    }
  }

  Texture3D(int64_t width, int64_t height, int64_t depth, int64_t channels, const torch::TensorOptions &options = {}) :
      _data(torch::empty({depth, height, width, channels}, options)) {}

  Texture3D(void *data_ptr, int64_t width, int64_t height, int64_t depth, int64_t channels,
            const torch::TensorOptions &options = {}) :
      _data(torch::from_blob(data_ptr, {depth, height, width, channels}, options).clone()) {}

  template<typename T>
    requires(std::is_fundamental_v<T>)
  Texture3D(const T &value, int64_t width, int64_t height, int64_t depth, int64_t channels,
            const torch::TensorOptions &options = {}) :
      _data(torch::full({depth, height, width, channels}, value, options)) {}

  static Texture3D cuda(int64_t width, int64_t height, int64_t depth, int64_t channels,
                        const torch::TensorOptions &options = {}) {
    Texture3D tex;
    tex._data = torch::empty({depth, height, width, channels}, options.device(torch::kCUDA));
    return tex;
  }

  static Texture3D cuda(void *data_ptr, int64_t width, int64_t height, int64_t depth, int64_t channels,
                        const torch::TensorOptions &options = {}) {
    Texture3D tex;
    if (!options.has_device() || options.device().is_cpu()) {
      tex._data = torch::from_blob(data_ptr, {depth, height, width, channels}, options).to(torch::kCUDA);
    } else if (options.device().is_cuda()) {
      tex._data = torch::from_blob(data_ptr, {depth, height, width, channels}, options).clone();
    } else {
      ERROR_AND_EXIT("Unsupported device type for Texture3D::cuda");
    }
    return tex;
  }

  template<typename T>
    requires(std::is_fundamental_v<T>)
  static Texture3D cuda(const T &value, int64_t width, int64_t height, int64_t depth, int64_t channels,
                        const torch::TensorOptions &options = {}) {
    Texture3D tex;
    tex._data = torch::full({depth, height, width, channels}, value, options.device(torch::kCUDA));
    return tex;
  }

  operator torch::Tensor() const { return _data; }

  const torch::Tensor &tensor() const { return _data; }

  torch::Tensor &tensor() { return _data; }

  int64_t width() const { return _data.size(2); }

  int64_t height() const { return _data.size(1); }

  int64_t depth() const { return _data.size(0); }

  int64_t channels() const { return _data.size(3); }

  bool empty() const { return _data.numel() == 0; }

  // uvw: [N, 3] with values in [-1, 1]
  torch::Tensor eval(const torch::Tensor &uvw, const TextureEvalParams &params = {}) const;

  void assign(const torch::Tensor &depths, const torch::Tensor &rows, const torch::Tensor &cols,
              const torch::Tensor &channels, const torch::Tensor &values) {
    // rows, cols, depths, channels, values: [N]
    _data.index_put_({depths, rows, cols, channels}, values);
  }


  void save_rawdata(const std::string &filename) const;

private:
  torch::Tensor _data;
};
